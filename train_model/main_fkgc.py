import os
import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import warnings

from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from fkgc_model import FKGCWithGAT, FKGCDataLoaderWithText
from dataloader import FKGCDataLoader

warnings.filterwarnings("ignore", message=".*AdamW.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")


def get_model_device(model):
    """Safely get model device, supports DataParallel"""
    if hasattr(model, 'module'):
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


def get_original_model(model):
    """Get original model, handles DataParallel wrapping"""
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class FKGC_Trainer:
    """FKGC model trainer"""

    def __init__(self, data_loader, model, tokenizer, optimizer, scheduler, device, hyperparams):
        self.data_loader = data_loader
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.hyperparams = hyperparams

        self.neg_sample_size = hyperparams.get('neg_sample_size', 3)

        self.model_device = get_model_device(self.model)

        if self.model_device != self.device:
            print(f"Warning: Model device ({self.model_device}) differs from trainer device ({self.device})")

    def _adjust_learning_rate(self, epoch):
        """Dynamically adjust learning rate"""
        if epoch <= 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.2
        elif epoch > 50:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.95

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        self._adjust_learning_rate(epoch)

        original_model = get_original_model(self.model)
        if hasattr(original_model, 'clear_prototype_cache'):
            original_model.clear_prototype_cache()

        train_relations = list(self.data_loader.train_tasks.keys())
        print(f"Training epoch {epoch}...")

        for rel_idx, query_rel in enumerate(train_relations):
            candidates = self.data_loader.rel2candidates.get(query_rel, [])
            candidates_id = [self.data_loader.ent2id[c] for c in candidates if c in self.data_loader.ent2id]

            if len(candidates_id) <= 20:
                continue

            rel_triples = self.data_loader.train_tasks[query_rel]
            train_tri_id = []
            for triple in rel_triples:
                h, r, t = triple
                if h in self.data_loader.ent2id and t in self.data_loader.ent2id:
                    train_tri_id.append([self.data_loader.ent2id[h], r, self.data_loader.ent2id[t]])

            train_tri_id = [trip for trip in train_tri_id if trip[0] != trip[2]]

            if len(train_tri_id) < self.data_loader.sp_num + 3:
                continue

            sampled_triples = random.sample(train_tri_id, self.data_loader.sp_num + 3)
            random.shuffle(sampled_triples)
            support_pair = sampled_triples[:self.data_loader.sp_num]
            query_pair_pos = sampled_triples[self.data_loader.sp_num:]

            support_pair_formatted = [[pair[0], pair[2]] for pair in support_pair]
            query_pair_pos_formatted = [[pair[0], pair[2]] for pair in query_pair_pos]

            support_triples = [(h_id, query_rel, t_id) for h_id, t_id in support_pair_formatted]
            query_triples = [(h_id, query_rel, t_id) for h_id, t_id in query_pair_pos_formatted]

            neg_triples = []
            for h_id, t_id in query_pair_pos_formatted:
                h, t = self.data_loader.id2ent[h_id], self.data_loader.id2ent[t_id]
                for _ in range(self.neg_sample_size):
                    neg_sample = self.data_loader.random_replacement(h, t, query_rel)
                    if neg_sample:
                        h_neg, r_neg, t_neg = neg_sample
                        if h_neg in self.data_loader.ent2id and t_neg in self.data_loader.ent2id:
                            neg_triples.append((self.data_loader.ent2id[h_neg], r_neg, self.data_loader.ent2id[t_neg]))

            if len(neg_triples) == 0:
                continue

            pos_scores = self.model(support_triples, query_triples, mode='training')
            neg_scores = self.model(support_triples, neg_triples, mode='training')

            margin = 1.0
            contrastive_loss = original_model.compute_margin_ranking_loss(pos_scores, neg_scores, margin)
            pos_mean = pos_scores.mean()
            neg_mean = neg_scores.mean()
            kl_reg = torch.relu(margin - (pos_mean - neg_mean))

            pos_l2_reg = torch.sqrt(torch.sum(pos_scores * pos_scores) + 1e-8)
            neg_l2_reg = torch.sqrt(torch.sum(neg_scores * neg_scores) + 1e-8)
            loss = contrastive_loss + 0.1 * kl_reg + 0.01 * pos_l2_reg + 0.01 * neg_l2_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, Total Batches: {batch_count}")
        return avg_loss

    def evaluate(self, split='valid', max_samples=None, max_relations=None):
        self.model.eval()

        if split == 'valid':
            tasks = self.data_loader.valid_tasks
        elif split == 'test':
            tasks = self.data_loader.test_tasks
        elif split == 'train':
            tasks = self.data_loader.train_tasks
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"\n=== Evaluating on {split} set ===")

        hit1_sum = []
        hit5_sum = []
        hit10_sum = []
        mrr_sum = []
        candidate_counts = []
        total_queries = 0

        import time
        start_time = time.time()
        batch_times = []

        original_model = get_original_model(self.model)
        if hasattr(original_model, 'cache_enabled'):
            original_model.cache_enabled = True
            if not hasattr(original_model, 'prototype_cache'):
                original_model.prototype_cache = {}

        use_amp = False
        scaler = None

        with torch.no_grad():
            for rel_idx, (relation, triples) in enumerate(tasks.items()):
                if max_relations and rel_idx >= max_relations:
                    break

                if len(triples) < self.data_loader.sp_num + 1:
                    continue

                candidates = self.data_loader.rel2candidates.get(relation, [])
                candidate_ent_id = [self.data_loader.ent2id[c] for c in candidates if c in self.data_loader.ent2id]

                if len(candidate_ent_id) == 0:
                    continue

                if len(triples) >= self.data_loader.sp_num + 1:
                    sampled_triples = random.sample(triples, len(triples))
                    random.shuffle(sampled_triples)
                    support_triples = sampled_triples[:self.data_loader.sp_num]
                    query_triples = sampled_triples[self.data_loader.sp_num:]
                else:
                    support_triples = triples[:self.data_loader.sp_num]
                    query_triples = triples[self.data_loader.sp_num:]

                if len(support_triples) == 0 or len(query_triples) == 0:
                    continue

                support_triples_id = []
                for h, r, t in support_triples:
                    if h in self.data_loader.ent2id and t in self.data_loader.ent2id:
                        support_triples_id.append((self.data_loader.ent2id[h], r, self.data_loader.ent2id[t]))

                if len(support_triples_id) == 0:
                    continue

                print(
                    f"Relation {rel_idx + 1}: {relation} - {len(support_triples)} support, {len(query_triples)} query (randomly sampled)")

                batch_size = 32
                query_batches = [query_triples[i:i + batch_size] for i in range(0, len(query_triples), batch_size)]
                
                relation_query_count = 0

                for batch_idx, query_batch in enumerate(query_batches):
                    if max_samples and total_queries >= max_samples:
                        break

                    batch_start_time = time.time()

                    batch_data = []
                    valid_queries = []

                    for query_triple in query_batch:
                        h, r, t = query_triple
                        if h not in self.data_loader.ent2id or t not in self.data_loader.ent2id:
                            continue

                        h_id = self.data_loader.ent2id[h]
                        t_id = self.data_loader.ent2id[t]

                        h_rel_key = h + r
                        correct_tails = self.data_loader.e1rel_e2.get(h_rel_key, [])
                        correct_tail_ids = [self.data_loader.ent2id[tail] for tail in correct_tails if
                                            tail in self.data_loader.ent2id]

                        negative_candidates = [c for c in candidate_ent_id if c not in correct_tail_ids]

                        if len(negative_candidates) == 0:
                            all_ent_ids = list(self.data_loader.ent2id.values())
                            negative_candidates = [c for c in all_ent_ids if c not in correct_tail_ids]

                        if len(negative_candidates) == 0:
                            continue

                        all_triples = [(h_id, r, neg_id) for neg_id in negative_candidates]
                        all_triples.append((h_id, r, t_id))

                        batch_data.append({
                            'triples': all_triples,
                            'h_id': h_id,
                            't_id': t_id,
                            'candidate_count': len(negative_candidates) + 1,
                            'correct_entities_count': len(correct_tail_ids)
                        })
                        
                        valid_queries.append(query_triple)

                    if not batch_data:
                        continue

                    all_triples_batch = [data['triples'] for data in batch_data]

                    query_triples_flat = []
                    for triples in all_triples_batch:
                        query_triples_flat.extend(triples)

                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            batch_scores = self.model(support_triples_id, query_triples_flat, mode='inference')
                    else:
                        batch_scores = self.model(support_triples_id, query_triples_flat, mode='inference')

                    if isinstance(batch_scores, tuple):
                        batch_scores = batch_scores[2]

                    score_idx = 0
                    for i, data in enumerate(batch_data):
                        num_candidates = len(data['triples'])
                        scores = batch_scores[score_idx:score_idx + num_candidates]
                        score_idx += num_candidates

                        if isinstance(scores, tuple):
                            scores = scores[2]

                        pos_score = scores[-1]
                        neg_scores = scores[:-1]
                        all_scores_combined = torch.cat([neg_scores, pos_score.unsqueeze(0)])
                        sorted_indices = torch.argsort(all_scores_combined, descending=True)
                        rank = (sorted_indices == len(neg_scores)).nonzero(as_tuple=True)[0].item() + 1

                        mrr_sum.append(1.0 / rank)
                        hit1_sum.append(1.0 if rank <= 1 else 0.0)
                        hit5_sum.append(1.0 if rank <= 5 else 0.0)
                        hit10_sum.append(1.0 if rank <= 10 else 0.0)
                        candidate_counts.append(data['candidate_count'])
                        total_queries += 1
                        relation_query_count += 1

                        if relation_query_count <= 3:
                            print(f"  Query {relation_query_count}: Rank {rank}/{data['candidate_count']} ")
                            
                        if total_queries % 10 == 0:
                            current_mrr = np.mean(mrr_sum) if mrr_sum else 0.0
                            current_hit1 = np.mean(hit1_sum) if hit1_sum else 0.0
                            current_hit5 = np.mean(hit5_sum) if hit5_sum else 0.0
                            current_hit10 = np.mean(hit10_sum) if hit10_sum else 0.0
                            import sys
                            sys.stdout.write(
                                f"Query #{total_queries}\tMRR: {current_mrr:.3f}\tHits@10: {current_hit10:.3f}\tHits@5: {current_hit5:.3f}\tHits@1: {current_hit1:.3f}\r")
                            sys.stdout.flush()


                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)


                if max_samples and total_queries >= max_samples:
                    break

        if total_queries > 0:
            print()

            total_time = time.time() - start_time
            avg_time_per_query = total_time / total_queries
            print(f"Evaluation completed in {total_time:.2f}s")
            print(f"Average time per query: {avg_time_per_query:.4f}s")
            print(f"Total queries processed: {total_queries}")

        if len(mrr_sum) > 0:
            avg_mrr = np.mean(mrr_sum)
            avg_hit1 = np.mean(hit1_sum)
            avg_hit5 = np.mean(hit5_sum)
            avg_hit10 = np.mean(hit10_sum)

            print(f"\n{split.upper()} Results:")
            print(f"  Total queries: {total_queries}")
            print(f"  MRR: {avg_mrr:.4f}")
            print(f"  Hits@1: {avg_hit1:.4f}")
            print(f"  Hits@5: {avg_hit5:.4f}")
            print(f"  Hits@10: {avg_hit10:.4f}")

            if candidate_counts:
                avg_candidates = np.mean(candidate_counts)
                min_candidates = np.min(candidate_counts)
                max_candidates = np.max(candidate_counts)
                print(f"  Candidate entities: avg={avg_candidates:.1f}, min={min_candidates}, max={max_candidates}")
        else:
            avg_mrr = avg_hit1 = avg_hit5 = avg_hit10 = 0.0
            print(f"\n{split.upper()} Results: No valid evaluations")

        return avg_mrr, avg_hit1, avg_hit5, avg_hit10

    def run(self):

        best_mrr = 0
        best_epoch = 0
        patience = 3
        no_improve_count = 0
        eval_interval = 3

        for epoch in range(self.hyperparams['epoch']):
            train_loss = self.train_epoch(epoch + 1)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

            should_evaluate = (epoch == 0) or ((epoch + 1) % eval_interval == 0)
            
            if should_evaluate:
                print(f"Evaluating epoch {epoch + 1}...")
                if getattr(self.data_loader, 'train_tasks', None) is not None and self.hyperparams.get('overfit_debug',
                                                                                                       False):
                    train_metrics = self.evaluate('train', max_samples=None, max_relations=None)
                valid_metrics = self.evaluate('valid', max_samples=None, max_relations=None)

                print(f"Epoch {epoch + 1} Results:")
                if self.hyperparams.get('overfit_debug', False):
                    print(
                        f"  Train(subset) - MRR: {train_metrics[0]:.4f}, Hits@1: {train_metrics[1]:.4f}, Hits@5: {train_metrics[2]:.4f}, Hits@10: {train_metrics[3]:.4f}")
                print(
                    f"  Valid - MRR: {valid_metrics[0]:.4f}, Hits@1: {valid_metrics[1]:.4f}, Hits@5: {valid_metrics[2]:.4f}, Hits@10: {valid_metrics[3]:.4f}")
                print("-" * 80)

                if valid_metrics[0] > best_mrr:
                    best_mrr = valid_metrics[0]
                    best_epoch = epoch + 1
                    no_improve_count = 0
                    self.save_model(f"best_fkgc_model_{self.hyperparams['identifier']}.pth")
                    print(f"New best model saved with Valid MRR: {valid_metrics[0]:.4f} (Epoch {best_epoch})")
                else:
                    no_improve_count += 1
                    print(f"No improvement for {no_improve_count} evaluations (current best: {best_mrr:.4f} at epoch {best_epoch})")
                
                if no_improve_count >= patience:
                    print(f"Early stopping triggered! No improvement for {patience} consecutive evaluations.")
                    print(f"Best model was at epoch {best_epoch} with MRR: {best_mrr:.4f}")
                    break

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_mrr': best_mrr,
                    'best_epoch': best_epoch,
                    'train_loss': train_loss,
                    'valid_metrics': valid_metrics
                }
                torch.save(checkpoint, f"checkpoint_{self.hyperparams['identifier']}.pth")

        print(f"Training completed! Best Valid MRR: {best_mrr:.4f} (achieved at epoch {best_epoch})")

        best_model_path = f"best_fkgc_model_{self.hyperparams['identifier']}.pth"
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final test evaluation...")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print("=" * 80)
            print("FINAL TEST EVALUATION WITH BEST MODEL")
            print("=" * 80)
            final_test_metrics = self.evaluate('test', max_samples=None, max_relations=None)
            print(f"Final Test Results (Best Model from Epoch {best_epoch}):")
            print(f"  MRR: {final_test_metrics[0]:.4f}")
            print(f"  Hits@1: {final_test_metrics[1]:.4f}")
            print(f"  Hits@5: {final_test_metrics[2]:.4f}")
            print(f"  Hits@10: {final_test_metrics[3]:.4f}")
            print("=" * 80)
        else:
            print("Warning: Best model file not found, using current model for final evaluation")

        print("Training and evaluation completed!")

        results_summary = {
            'best_valid_mrr': best_mrr
        }

        results_file = f"results_{self.hyperparams['identifier']}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Best Valid MRR: {best_mrr:.4f}\n")

        print(f"Results saved to {results_file}")
        print("Training and evaluation completed!")

    def save_model(self, save_path):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparams': self.hyperparams
        }

        if self.hyperparams.get('use_lora', False):
            from lora_utils import save_lora_weights
            lora_save_path = save_path.replace('.pth', '_lora.pth')
            save_lora_weights(self.model, lora_save_path)
            print(f"LoRA weights saved to {lora_save_path}")

        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--bert_lr', type=float, default=5e-5)
    parser.add_argument('--model_lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--data', type=str, default='fb15k-237')
    parser.add_argument('--plm', type=str, default='bert', choices=['bert', 'bert_tiny', 'roberta'])
    parser.add_argument('--max_desc_length', type=int, default=128)
    parser.add_argument('--sp_num', type=int, default=5)
    parser.add_argument('--gat_heads', type=int, default=8)
    parser.add_argument('--gat_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--max_subgraph_edges', type=int, default=20000)
    parser.add_argument('--use_gat', default=True, action='store_true')
    parser.add_argument('--no_gat', dest='use_gat', action='store_false')
    parser.add_argument('--fine_tune_bert', default=True, action='store_true')
    parser.add_argument('--scoring_paradigm', type=str, default='transe', choices=['transe', 'fkgc'],
                        help='Scoring paradigm: transe (TransE) or fkgc (FKGC with relation prototypes)')
    parser.add_argument('--fkgc_attention_temp', type=float, default=1.0)
    parser.add_argument('--fkgc_prototype_shrinkage', type=float, default=0.2)
    parser.add_argument('--fkgc_attention_weight', type=float, default=0.8)
    parser.add_argument('--use_lora', default=True, action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,v_proj,k_proj,o_proj')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['constant', 'cosine', 'linear'])
    parser.add_argument('--task', default='FKGC', choices=['LP', 'TC', 'FKGC'])
    parser.add_argument('--neg_sample_size', type=int, default=3)
    parser.add_argument('--overfit_debug', action='store_true', default=False)
    parser.add_argument('--overfit_relations', type=int, default=1)
    parser.add_argument('--overfit_queries', type=int, default=5)

    arg = parser.parse_args()


if arg.use_gat:
    if arg.use_lora:
        identifier = '{}-{}-fkgc-batch_size={}-sp_num={}-gat_heads={}-gat_layers={}-GAT-lora_r={}-lora_alpha={}'.format(
            arg.data, arg.plm, arg.batch_size, arg.sp_num,
            arg.gat_heads, arg.gat_layers, arg.lora_r, arg.lora_alpha)
    elif arg.fine_tune_bert:
        identifier = '{}-{}-fkgc-batch_size={}-sp_num={}-gat_heads={}-gat_layers={}-GAT-FT_BERT'.format(
            arg.data, arg.plm, arg.batch_size, arg.sp_num,
            arg.gat_heads, arg.gat_layers)
    else:
        identifier = '{}-{}-fkgc-batch_size={}-sp_num={}-gat_heads={}-gat_layers={}-GAT'.format(
            arg.data, arg.plm, arg.batch_size, arg.sp_num,
            arg.gat_heads, arg.gat_layers)
else:
    if arg.use_lora:
        identifier = '{}-{}-fkgc-batch_size={}-sp_num={}-mlp_layers={}-NO_GAT-lora_r={}-lora_alpha={}'.format(
            arg.data, arg.plm, arg.batch_size, arg.sp_num,
            arg.gat_layers, arg.lora_r, arg.lora_alpha)
    elif arg.fine_tune_bert:
        identifier = '{}-{}-fkgc-batch_size={}-sp_num={}-mlp_layers={}-NO_GAT-FT_BERT'.format(
            arg.data, arg.plm, arg.batch_size, arg.sp_num,
            arg.gat_layers)
    else:
        identifier = '{}-{}-fkgc-batch_size={}-sp_num={}-mlp_layers={}-NO_GAT'.format(
            arg.data, arg.plm, arg.batch_size, arg.sp_num,
            arg.gat_layers)

random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

if arg.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpus
    print(f"Set CUDA_VISIBLE_DEVICES to: {arg.gpus}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

if arg.plm == 'bert':
    plm_name = "bert"
    t_model = 'bert'
elif arg.plm == 'bert_tiny':
    plm_name = "prajjwal1/bert-tiny"
    t_model = 'bert'
elif arg.plm == 'roberta':
    plm_name = "roberta-base"
    t_model = 'roberta'
local_cache_path = '../../.cache/bert'
print(f"Using cache path: {local_cache_path}")

if arg.data == 'nell':
    in_paths = {
        'dataset': arg.data,
        'train': './data/nell/train_tasks.tsv',
        'valid': './data/nell/dev_tasks.tsv',
        'test': './data/nell/test_tasks.tsv',
        'text': ['./data/nell/mycotdes.txt',
                 './data/nell/relation_use.txt'],
        'ent2ids': './data/nell/ent2ids.txt',
        'transE_entity2id': './data/nell/ent2ids.txt',
        'transE_embedding': './data/nell/entity2vec.TransE',
        'rel2cands': './data/nell/rel2cands.json',
        'e1rel_e2': './data/nell/e1rel_e2.json'
    }
elif arg.data == 'fb15k237-one':
    in_paths = {
        'dataset': arg.data,
        'train': './data/fb15k237-one/train_tasks.tsv',
        'valid': './data/fb15k237-one/dev_tasks.tsv',
        'test': './data/fb15k237-one/test_tasks.tsv',
        'text': ['./data/fb15k237-one/cotdes.txt',
                 './data/fb15k237-one/relation2text.txt'],
        'ent2ids': './data/fb15k237-one/ent2ids.txt',
        'ent2type': './data/fb15k237-one/entity2type.txt',
        'transE_entity2id': './data/fb15k237-one/ent2ids.txt',
        'transE_embedding': './data/fb15k237-one/entity2vec.TransE',
        'rel2cands': './data/fb15k237-one/rel2cands.json',
        'e1rel_e2': './data/fb15k237-one/e1rel_e2.json'
    }

lm_config = AutoConfig.from_pretrained(local_cache_path)
lm_tokenizer = AutoTokenizer.from_pretrained(local_cache_path, do_basic_tokenize=False)

data_loader = FKGCDataLoaderWithText(
    in_paths=in_paths,
    tokenizer=lm_tokenizer,
    batch_size=arg.batch_size,
    max_desc_length=arg.max_desc_length,
    model=t_model,
    sp_num=arg.sp_num
)

data_loader.transE_entity2id_path = in_paths.get('transE_entity2id')
data_loader.transE_embedding_path = in_paths.get('transE_embedding')

if data_loader.transE_entity2id_path and data_loader.transE_embedding_path:
    data_loader.load_transE_embeddings(data_loader.transE_entity2id_path, data_loader.transE_embedding_path)
    
    if hasattr(data_loader, 'transE_embeddings') and data_loader.transE_embeddings is not None:
        data_loader.set_transE_structural_embeddings(data_loader.transE_embeddings, data_loader.entity_to_id)

print(f"Number of entities: {len(data_loader.entity_set)}")
print(f"Number of relations: {len(data_loader.relation_set)}")

data_loader.neg_sample_size = arg.neg_sample_size

data_loader.print_negative_sampling_stats()

try:
    preview_relations = list(data_loader.train_tasks.keys())[:2]
    for rel in preview_relations:
        rel_triples = data_loader.train_tasks.get(rel, [])
        if len(rel_triples) < max(3, data_loader.sp_num + 1):
            continue

        # Support set and query set (converted to ID pairs)
        support_pair = [[data_loader.ent2id[h], data_loader.ent2id[t]] for (h, _, t) in
                        rel_triples[:data_loader.sp_num]]
        query_pair_pos = [[data_loader.ent2id[h], data_loader.ent2id[t]] for (h, _, t) in
                          rel_triples[data_loader.sp_num:data_loader.sp_num + 3]]

        # Generate negative samples (returns entity name triples)
        neg_samples = data_loader.generate_negative_samples(support_pair, query_pair_pos, rel, num_negatives=min(5,
                                                                                                                 getattr(
                                                                                                                     data_loader,
                                                                                                                     'neg_sample_size',
                                                                                                                     3)))

        print("\n=== Negative sampling preview ===")
        print(f"Relation: {rel}")
        # Display one positive sample
        pos_h_id, pos_t_id = query_pair_pos[0]
        pos_h = data_loader.id2ent[pos_h_id]
        pos_t = data_loader.id2ent[pos_t_id]
        print(f"Positive: ({pos_h}, {rel}, {pos_t})")
        # Display several negative samples
        for i, neg in enumerate(neg_samples[:5]):
            nh, nr, nt = neg
            print(f"Negative {i + 1}: ({nh}, {nr}, {nt})")
except Exception as e:
    print(f"Warning: failed to preview negative samples: {e}")


lora_config = None
if arg.use_lora:
    from lora_utils import LoRAConfig

    target_modules = arg.lora_target_modules.split(',')
    lora_config = LoRAConfig(
        r=arg.lora_r,
        lora_alpha=arg.lora_alpha,
        lora_dropout=arg.lora_dropout,
        target_modules=target_modules
    )
    print(f"LoRA config: r={arg.lora_r}, alpha={arg.lora_alpha}, dropout={arg.lora_dropout}")
    print(f"LoRA target modules: {target_modules}")

# print("Initializing FKGC model...")
model = FKGCWithGAT(
    bert_model_name=local_cache_path,
    n_ent=len(data_loader.entity_set),
    n_rel=len(data_loader.relation_set),
    hidden_dim=arg.hidden_dim,
    gat_heads=arg.gat_heads,
    gat_layers=arg.gat_layers,
    dropout=0.1,
    use_lora=arg.use_lora,
    lora_config=lora_config,
    data_loader=data_loader,
    max_subgraph_edges=arg.max_subgraph_edges,
    use_gat=arg.use_gat,
    debug_mode=False,
    fine_tune_bert=arg.fine_tune_bert,
    scoring_paradigm=arg.scoring_paradigm,
    fkgc_attention_temp=arg.fkgc_attention_temp,
    fkgc_prototype_shrinkage=arg.fkgc_prototype_shrinkage,
    fkgc_attention_weight=arg.fkgc_attention_weight
)

model = model.to(device)
print(f"Model moved to {device}")

print("Initializing entity embeddings from TransE structural embeddings...")
try:
    if hasattr(data_loader, 'transE_loaded') and data_loader.transE_loaded:
        print("Using TransE structural embeddings from dataloader...")
        
        transE_embeddings = data_loader.structural_entity_embeddings
        transE_entity2id = data_loader.transE_entity2id
        
        if transE_embeddings is not None and transE_entity2id is not None:
            weight_device = model.ent_embeddings.weight.device
            weight_dtype = model.ent_embeddings.weight.dtype
            hidden_dim = model.ent_embeddings.weight.size(-1)
            
            num_entities = len(data_loader.ent2id)
            aligned = torch.zeros((num_entities, hidden_dim), dtype=weight_dtype)
            
            transe_matrix = torch.from_numpy(transE_embeddings).float()
            transe_dim = transe_matrix.size(-1)
            
            loaded_count = 0
            for ent_name, ent_idx in data_loader.ent2id.items():
                if ent_name in transE_entity2id:
                    transe_id = transE_entity2id[ent_name]
                    vec = transe_matrix[transe_id]
                    if transe_dim >= hidden_dim:
                        aligned[ent_idx] = vec[:hidden_dim]
                    else:
                        temp = torch.zeros(hidden_dim)
                        temp[:transe_dim] = vec
                        aligned[ent_idx] = temp
                    loaded_count += 1
            
            model.ent_embeddings.weight.data.copy_(aligned.to(weight_device))
            print(f"Initialized entity embeddings from TransE: {loaded_count} entities, dim {hidden_dim} (source dim {transe_dim})")
        else:
            print("TransE embeddings not available, using random initialization")
    else:
        print("TransE not loaded in dataloader, using random initialization")
except Exception as e:
    print(f"Warning: failed to initialize entity embeddings from TransE: {e}")
    print("Using random initialization")

entity_texts = data_loader.get_entity_texts()

if entity_texts:
    print(f"Found {len(entity_texts)} entity texts for BERT initialization")
    sample_entities = list(entity_texts.items())[:3]
    for ent_id, text in sample_entities:
        print(f"  Entity {ent_id}: {text[:50]}...")
    print("Starting BERT initialization for semantic embeddings...")
    print("Initializing entity embeddings with BERT...")
    model.initialize_entity_embeddings(entity_texts)
    model.initialize_semantic_entity_embeddings(entity_texts)
    model.use_transe_for_semantic = False
    print("BERT initialization completed!")
    
    bert_embeddings = model.ent_embeddings.weight.data.cpu()
    data_loader.set_bert_entity_embeddings(bert_embeddings)
else:
    print("Warning: No entity texts found, skipping BERT initialization")
    print("This might be because:")
    print("1. Text files are not loaded properly")
    print("2. Entity mapping is not correct")
    print("3. Text files don't contain entity descriptions")

train_triples = []
for h, r, t in data_loader.train_set:
    if h in data_loader.ent2id and r in data_loader.rel2id and t in data_loader.ent2id:
        train_triples.append((data_loader.ent2id[h], data_loader.rel2id[r], data_loader.ent2id[t]))
entity_embeddings = get_original_model(model).ent_embeddings.weight.data
model.initialize_relation_embeddings(train_triples, entity_embeddings)

gpu_count = torch.cuda.device_count()

if gpu_count > 1:
    print(f"Using single GPU: {device}")
else:
    print(f"Using single GPU: {device}")
    print("DataParallel not enabled (only 1 GPU available)")

no_decay = ["bias", "LayerNorm.weight"]

if arg.use_lora:
    from lora_utils import get_lora_parameters

    lora_params = get_lora_parameters(model)
    param_group = [
        {'lr': arg.model_lr, 'params': lora_params, 'weight_decay': arg.weight_decay},
    ]

    non_lora_params = [p for n, p in model.named_parameters()
                       if 'lora_A' not in n and 'lora_B' not in n]
    param_group += [
        {'lr': arg.model_lr, 'params': non_lora_params, 'weight_decay': arg.weight_decay},
    ]
else:
    param_group = [
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('bert' not in n) and
                                        (not any(nd in n for nd in no_decay))],
         'weight_decay': arg.weight_decay},
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('bert' not in n) and
                                        (any(nd in n for nd in no_decay))],
         'weight_decay': 0.0},
    ]

    if arg.fine_tune_bert:
        param_group += [
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('bert' in n) and
                                           (not any(nd in n for nd in no_decay))],
             'weight_decay': arg.weight_decay},
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('bert' in n) and
                                           (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0},
        ]

optimizer = AdamW(param_group)

total_steps = arg.epoch * 1000

if arg.scheduler_type == 'cosine':
    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"Using PyTorch CosineAnnealingLR with T_max={total_steps}")
elif arg.scheduler_type == 'linear':
    from torch.optim.lr_scheduler import LambdaLR

    lr_lambda = lambda step: 1.0 - step / max(1, total_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Using PyTorch LambdaLR linear decay over {total_steps} steps")
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    print("Using constant learning rate (no scheduler, no warmup)")

hyperparams = {
    'batch_size': arg.batch_size,
    'epoch': arg.epoch,
    'identifier': identifier,
    'evaluate_every': 1,
    'update_every': 1,

    'plm': arg.plm,
    'max_desc_length': arg.max_desc_length,
    'task': arg.task,
    'sp_num': arg.sp_num,
    'gat_heads': arg.gat_heads,
    'gat_layers': arg.gat_layers,
    'hidden_dim': arg.hidden_dim,
    'use_gat': arg.use_gat,
    'use_lora': arg.use_lora,
    'lora_r': arg.lora_r,
    'lora_alpha': arg.lora_alpha,
    'lora_dropout': arg.lora_dropout,
    'fine_tune_bert': arg.fine_tune_bert,  
    'neg_sample_size': arg.neg_sample_size,
    'max_subgraph_edges': arg.max_subgraph_edges,
    'scheduler_type': arg.scheduler_type,
    'bert_lr': arg.bert_lr,  
    'model_lr': arg.model_lr,  
    'overfit_debug': arg.overfit_debug
}

if arg.overfit_debug:
    print("Overfit debug mode enabled: restricting tasks to tiny subset")


    def _restrict_tasks(tasks_dict, max_relations, max_queries_per_rel):
        new_tasks = {}
        for idx, (rel, triples) in enumerate(tasks_dict.items()):
            if idx >= max_relations:
                break
            if len(triples) >= max(1, data_loader.sp_num + 1):
                kept = triples[:data_loader.sp_num + max_queries_per_rel]
                new_tasks[rel] = kept
        return new_tasks


    data_loader.train_tasks = _restrict_tasks(data_loader.train_tasks, arg.overfit_relations, arg.overfit_queries)
    data_loader.valid_tasks = _restrict_tasks(data_loader.valid_tasks, arg.overfit_relations, arg.overfit_queries)
    data_loader.test_tasks = _restrict_tasks(data_loader.test_tasks, arg.overfit_relations, arg.overfit_queries)
    print(
        f"Overfit subset sizes -> train:{len(data_loader.train_tasks)}, valid:{len(data_loader.valid_tasks)}, test:{len(data_loader.test_tasks)}")

trainer = FKGC_Trainer(data_loader, model, lm_tokenizer, optimizer, scheduler, device, hyperparams)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.run()
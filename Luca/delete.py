import numpy as np                                              

data = np.load('hidden_states_Qwen_Qwen2.5-VL-3B-Instruct_all_last.npz', allow_pickle=True)            

print('=== SHAPES ===')                                                                                
for key in data.files:                                          
  print(f'  {key:10s}  {data[key].shape}  {data[key].dtype}')                                        
                                                                                                     
print('\n=== LABEL DISTRIBUTION ===')
labels = data['labels']                                                                                
print(f'  factual   (0): {(labels==0).sum()}  ({100*(labels==0).mean():.1f}%)')
print(f'  nonfactual(1): {(labels==1).sum()}  ({100*(labels==1).mean():.1f}%)')                        
                                                                                                     
print('\n=== SPLIT DISTRIBUTION ===')                                                                  
for s in ['train_val', 'test']:                                                                        
  print(f'  {s}: {(data["splits"]==s).sum()}')                                                       
                                                                                                     
print('\n=== SAMPLE PREDICTIONS ===')                                                                  
for i in range(5):                                                                                     
  print(f'  [{i}] pred={data["preds"][i]:5s}  gt={data["gts"][i]:3s}  label={labels[i]}')
                                                                                                     
If you're in Colab and the variables are still in memory from the collection run, you don't even need  
to reload — just run:                                                                                  
                                                                                                     
print(all_features.shape)   # (N, layers, hidden_dim)           
print(all_labels[:10])

####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
multi_Z_A GoogleColab
claude-af
python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/2LHC/2LHC.fasta,/content/jaxalpha/multi/2LHG/2LHG.fasta,/content/jaxalpha/multi/2LHE/2LHE.fasta,/content/jaxalpha/multi/2LHD/2LHD.fasta,/content/jaxalpha/multi/6UF2/6UF2.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/1_5 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/2LHC/feature.pkl,/content/jaxalpha/multi/2LHG/feature.pkl,/content/jaxalpha/multi/2LHE/feature.pkl,/content/jaxalpha/multi/2LHD/feature.pkl,/content/jaxalpha/multi/6UF2/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true


python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/7CN6/7CN6.fasta,/content/jaxalpha/multi/7JTL/7JTL.fasta,/content/jaxalpha/multi/7CWP/7CWP.fasta,/content/jaxalpha/multi/6Y4F/6Y4F.fasta,/content/jaxalpha/multi/6ZYC/6ZYC.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/6_10 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/7CN6/feature.pkl,/content/jaxalpha/multi/7JTL/feature.pkl,/content/jaxalpha/multi/7CWP/feature.pkl,/content/jaxalpha/multi/6Y4F/feature.pkl,/content/jaxalpha/multi/6ZYC/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true



python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/8EM5/8EM5.fasta,/content/jaxalpha/multi/8PBV/8PBV.fasta,/content/jaxalpha/multi/7ROA/7ROA.fasta,/content/jaxalpha/multi/7PZT/7PZT.fasta,/content/jaxalpha/multi/8D27/8D27.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/11_15 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/8EM5/feature.pkl,/content/jaxalpha/multi/8PBV/feature.pkl,/content/jaxalpha/multi/7ROA/feature.pkl,/content/jaxalpha/multi/7PZT/feature.pkl,/content/jaxalpha/multi/8D27/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true
  
  
####-------------------------------------------------------------------------------------------
multi_A_Z GoogleColab 
claude-af

python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/6UF2/6UF2.fasta,/content/jaxalpha/multi/2LHG/2LHG.fasta,/content/jaxalpha/multi/2LHE/2LHE.fasta,/content/jaxalpha/multi/2LHD/2LHD.fasta,/content/jaxalpha/multi/2LHC/2LHC.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_A_Z/1_5 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/6UF2/feature.pkl,/content/jaxalpha/multi/2LHG/feature.pkl,/content/jaxalpha/multi/2LHE/feature.pkl,/content/jaxalpha/multi/2LHD/feature.pkl,/content/jaxalpha/multi/2LHC/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true

python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/6ZYC/6ZYC.fasta,/content/jaxalpha/multi/6Y4F/6Y4F.fasta,/content/jaxalpha/multi/7CWP/7CWP.fasta,/content/jaxalpha/multi/7JTL/7JTL.fasta,/content/jaxalpha/multi/7CN6/7CN6.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_A_Z/6_10 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/6ZYC/feature.pkl,/content/jaxalpha/multi/6Y4F/feature.pkl,/content/jaxalpha/multi/7CWP/feature.pkl,/content/jaxalpha/multi/7JTL/feature.pkl,/content/jaxalpha/multi/7CN6/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true


python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/8D27/8D27.fasta,/content/jaxalpha/multi/7PZT/7PZT.fasta,/content/jaxalpha/multi/7ROA/7ROA.fasta,/content/jaxalpha/multi/8PBV/8PBV.fasta,/content/jaxalpha/multi/8EM5/8EM5.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_A_Z/11_15 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/8D27/feature.pkl,/content/jaxalpha/multi/7PZT/feature.pkl,/content/jaxalpha/multi/7ROA/feature.pkl,/content/jaxalpha/multi/8PBV/feature.pkl,/content/jaxalpha/multi/8EM5/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true

####-------------------------------------------------------------------------------------------
multi_same GoogleColab 
claude-af

python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/2LHG/2LHG.fasta,/content/jaxalpha/multi/2LHE/2LHE.fasta,/content/jaxalpha/multi/2LHD/2LHD.fasta,/content/jaxalpha/multi/2LHC/2LHC.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_same \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/2LHG/feature.pkl,/content/jaxalpha/multi/2LHE/feature.pkl,/content/jaxalpha/multi/2LHD/feature.pkl,/content/jaxalpha/multi/2LHC/feature.pkl \
  --use_gpu_relax=true
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------

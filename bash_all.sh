GoogleColab_descend_all

python parjax_cla_multi_gpu0.py \
  --fasta_paths=/content/jaxalpha/multi/8D27/8D27.fasta,/content/jaxalpha/multi/7PZT/7PZT.fasta,/content/jaxalpha/multi/6ZYC/6ZYC.fasta,/content/jaxalpha/multi/6Y4F/6Y4F.fasta,/content/jaxalpha/multi/7CWP/7CWP.fasta,/content/jaxalpha/multi/7ROA/7ROA.fasta,/content/jaxalpha/multi/6UF2/6UF2.fasta,/content/jaxalpha/multi/8PBV/8PBV.fasta,/content/jaxalpha/multi/7JTL/7JTL.fasta,/content/jaxalpha/multi/8EM5/8EM5.fasta,/content/jaxalpha/multi/7CN6/7CN6.fasta,/content/jaxalpha/multi/2LHC/2LHC.fasta,/content/jaxalpha/multi/2LHG/2LHG.fasta,/content/jaxalpha/multi/2LHE/2LHE.fasta,/content/jaxalpha/multi/2LHD/2LHD.fasta   \
  --output_dir=/content/multi \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/8D27/feature.pkl,/content/jaxalpha/multi/7PZT/feature.pkl,/content/jaxalpha/multi/6ZYC/feature.pkl,/content/jaxalpha/multi/6Y4F/feature.pkl,/content/jaxalpha/multi/7CWP/feature.pkl,/content/jaxalpha/multi/7ROA/feature.pkl,/content/jaxalpha/multi/6UF2/feature.pkl,/content/jaxalpha/multi/8PBV/feature.pkl,/content/jaxalpha/multi/7JTL/feature.pkl,/content/jaxalpha/multi/8EM5/feature.pkl,/content/jaxalpha/multi/7CN6/feature.pkl,/content/jaxalpha/multi/2LHC/feature.pkl,/content/jaxalpha/multi/2LHG/feature.pkl,/content/jaxalpha/multi/2LHE/feature.pkl,/content/jaxalpha/multi/2LHD/feature.pkl  \
  --use_gpu_relax=true
  



####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
123_descend_all   batch

python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/8D27/8D27.fasta,/home/koohi/fereidoon/ParallelFold/multi/7PZT/7PZT.fasta,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/6ZYC.fasta,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/6Y4F.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CWP/7CWP.fasta,/home/koohi/fereidoon/ParallelFold/multi/7ROA/7ROA.fasta,/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta,/home/koohi/fereidoon/ParallelFold/multi/8PBV/8PBV.fasta,/home/koohi/fereidoon/ParallelFold/multi/7JTL/7JTL.fasta,/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta   \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/batch/gpu0/  \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/8D27/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7PZT/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CWP/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7ROA/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8PBV/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7JTL/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl  \
  --use_gpu_relax=true 
  

python parjax_cla_multi.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/8D27/8D27.fasta,/home/koohi/fereidoon/ParallelFold/multi/7PZT/7PZT.fasta,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/6ZYC.fasta,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/6Y4F.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CWP/7CWP.fasta,/home/koohi/fereidoon/ParallelFold/multi/7ROA/7ROA.fasta,/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta,/home/koohi/fereidoon/ParallelFold/multi/8PBV/8PBV.fasta,/home/koohi/fereidoon/ParallelFold/multi/7JTL/7JTL.fasta,/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta   \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/batch/gpu1/  \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/8D27/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7PZT/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CWP/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7ROA/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8PBV/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7JTL/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl  \
  --use_gpu_relax=true 


####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------
123_descend_all   benchmark

python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/8D27/8D27.fasta,/home/koohi/fereidoon/ParallelFold/multi/7PZT/7PZT.fasta,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/6ZYC.fasta,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/6Y4F.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CWP/7CWP.fasta,/home/koohi/fereidoon/ParallelFold/multi/7ROA/7ROA.fasta,/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta,/home/koohi/fereidoon/ParallelFold/multi/8PBV/8PBV.fasta,/home/koohi/fereidoon/ParallelFold/multi/7JTL/7JTL.fasta,/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta   \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/benchmark/gpu0/  \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/8D27/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7PZT/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CWP/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7ROA/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8PBV/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7JTL/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl  \
  --use_gpu_relax=true \
  --benchmark=true
  

python parjax_cla_multi.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/8D27/8D27.fasta,/home/koohi/fereidoon/ParallelFold/multi/7PZT/7PZT.fasta,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/6ZYC.fasta,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/6Y4F.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CWP/7CWP.fasta,/home/koohi/fereidoon/ParallelFold/multi/7ROA/7ROA.fasta,/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta,/home/koohi/fereidoon/ParallelFold/multi/8PBV/8PBV.fasta,/home/koohi/fereidoon/ParallelFold/multi/7JTL/7JTL.fasta,/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta   \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/benchmark/gpu1/  \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/8D27/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7PZT/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CWP/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7ROA/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8PBV/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7JTL/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl  \
  --use_gpu_relax=true \
  --benchmark=true




wwwwwwwwwwwwwwwwwwwww



python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/test \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl \
  --use_gpu_relax=true \
  --benchmark=true
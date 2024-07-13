####-------------------------------------------------------------------------------------------
multi_Z_A GoogleColab
claude-af
python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/2LHC/2LHC.fasta,/content/jaxalpha/multi/2LHG/2LHG.fasta,/content/jaxalpha/multi/2LHE/2LHE.fasta,/content/jaxalpha/multi/2LHD/2LHD.fasta,/content/jaxalpha/multi/6UF2/6UF2.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/1_5 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/2LHC/feature.pkl,/content/jaxalpha/multi/2LHG/feature.pkl,/content/jaxalpha/multi/2LHE/feature.pkl,/content/jaxalpha/multi/2LHD/feature.pkl,/content/jaxalpha/multi/6UF2/feature.pkl \
  --use_gpu_relax=true


python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/7CN6/7CN6.fasta,/content/jaxalpha/multi/7JTL/7JTL.fasta,/content/jaxalpha/multi/7CWP/7CWP.fasta,/content/jaxalpha/multi/6Y4F/6Y4F.fasta,/content/jaxalpha/multi/6ZYC/6ZYC.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/6_10 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/7CN6/feature.pkl,/content/jaxalpha/multi/7JTL/feature.pkl,/content/jaxalpha/multi/7CWP/feature.pkl,/content/jaxalpha/multi/6Y4F/feature.pkl,/content/jaxalpha/multi/6ZYC/feature.pkl \
  --use_gpu_relax=true



python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/8EM5/8EM5.fasta,/content/jaxalpha/multi/8PBV/8PBV.fasta,/content/jaxalpha/multi/7ROA/7ROA.fasta,/content/jaxalpha/multi/7PZT/7PZT.fasta,/content/jaxalpha/multi/8D27/8D27.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/11_15 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/8EM5/feature.pkl,/content/jaxalpha/multi/8PBV/feature.pkl,/content/jaxalpha/multi/7ROA/feature.pkl,/content/jaxalpha/multi/7PZT/feature.pkl,/content/jaxalpha/multi/8D27/feature.pkl \
  --use_gpu_relax=true
  
  
####-------------------------------------------------------------------------------------------
multi_A_Z GoogleColab 
claude-af

python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/2LHC/2LHC.fasta,/content/jaxalpha/multi/2LHG/2LHG.fasta,/content/jaxalpha/multi/2LHE/2LHE.fasta,/content/jaxalpha/multi/2LHD/2LHD.fasta,/content/jaxalpha/multi/6UF2/6UF2.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/1_5 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/2LHC/feature.pkl,/content/jaxalpha/multi/2LHG/feature.pkl,/content/jaxalpha/multi/2LHE/feature.pkl,/content/jaxalpha/multi/2LHD/feature.pkl,/content/jaxalpha/multi/6UF2/feature.pkl \
  --use_gpu_relax=true


python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/7CN6/7CN6.fasta,/content/jaxalpha/multi/7JTL/7JTL.fasta,/content/jaxalpha/multi/7CWP/7CWP.fasta,/content/jaxalpha/multi/6Y4F/6Y4F.fasta,/content/jaxalpha/multi/6ZYC/6ZYC.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/6_10 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/7CN6/feature.pkl,/content/jaxalpha/multi/7JTL/feature.pkl,/content/jaxalpha/multi/7CWP/feature.pkl,/content/jaxalpha/multi/6Y4F/feature.pkl,/content/jaxalpha/multi/6ZYC/feature.pkl \
  --use_gpu_relax=true



python parjax_cla_multi_GoogleColab.py \
  --fasta_paths=/content/jaxalpha/multi/8EM5/8EM5.fasta,/content/jaxalpha/multi/8PBV/8PBV.fasta,/content/jaxalpha/multi/7ROA/7ROA.fasta,/content/jaxalpha/multi/7PZT/7PZT.fasta,/content/jaxalpha/multi/8D27/8D27.fasta  \
  --output_dir=/content/jaxalpha/multi/out_multi_Z_A/11_15 \
  --parameter_path=/content/ParallelFold/params \
  --feature_files=/content/jaxalpha/multi/8EM5/feature.pkl,/content/jaxalpha/multi/8PBV/feature.pkl,/content/jaxalpha/multi/7ROA/feature.pkl,/content/jaxalpha/multi/7PZT/feature.pkl,/content/jaxalpha/multi/8D27/feature.pkl \
  --use_gpu_relax=true

####-------------------------------------------------------------------------------------------
multi_A_Z
claude-af
python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_A_Z/1_5 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl \
  --use_gpu_relax=true


python parjax_cla_multi.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/6ZYC/6ZYC.fasta,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/6Y4F.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CWP/7CWP.fasta,/home/koohi/fereidoon/ParallelFold/multi/7JTL/7JTL.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_A_Z/6_10 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/6ZYC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CWP/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7JTL/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl \
  --use_gpu_relax=true



python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/8D27/8D27.fasta,/home/koohi/fereidoon/ParallelFold/multi/7PZT/7PZT.fasta,/home/koohi/fereidoon/ParallelFold/multi/7ROA/7ROA.fasta,/home/koohi/fereidoon/ParallelFold/multi/8PBV/8PBV.fasta,/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_A_Z/11_15 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/8D27/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7PZT/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7ROA/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8PBV/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl \
  --use_gpu_relax=true

####-------------------------------------------------------------------------------------------
multi_Z_A
claude-af
python parjax_cla_multi.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta,/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_Z_A/1_5 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl \
  --use_gpu_relax=true


python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/7CN6/7CN6.fasta,/home/koohi/fereidoon/ParallelFold/multi/7JTL/7JTL.fasta,/home/koohi/fereidoon/ParallelFold/multi/7CWP/7CWP.fasta,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/6Y4F.fasta,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/6ZYC.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_Z_A/6_10 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/7CN6/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7JTL/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7CWP/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6Y4F/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6ZYC/feature.pkl \
  --use_gpu_relax=true



python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/8EM5/8EM5.fasta,/home/koohi/fereidoon/ParallelFold/multi/8PBV/8PBV.fasta,/home/koohi/fereidoon/ParallelFold/multi/7ROA/7ROA.fasta,/home/koohi/fereidoon/ParallelFold/multi/7PZT/7PZT.fasta,/home/koohi/fereidoon/ParallelFold/multi/8D27/8D27.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_Z_A/11_15 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/8EM5/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8PBV/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7ROA/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/7PZT/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/8D27/feature.pkl \
  --use_gpu_relax=true


####-------------------------------------------------------------------------------------------
multi_same
claude-af
python parjax_cla_multi.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_same/out_multi_same_gpu1 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl \
  --use_gpu_relax=true


claude-af
python parjax_cla_multi_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/multi/2LHC/2LHC.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHG/2LHG.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHE/2LHE.fasta,/home/koohi/fereidoon/ParallelFold/multi/2LHD/2LHD.fasta,/home/koohi/fereidoon/ParallelFold/multi/6UF2/6UF2.fasta  \
  --output_dir=/home/koohi/fereidoon/ParallelFold/multi/out_multi_same/out_multi_same_gpu0 \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_files=/home/koohi/fereidoon/ParallelFold/multi/2LHC/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHG/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHE/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/2LHD/feature.pkl,/home/koohi/fereidoon/ParallelFold/multi/6UF2/feature.pkl \
  --use_gpu_relax=true

####-------------------------------------------------------------------------------------------
gpt-AF
python parjax_gp.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/input/mono_set1/8PBV.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/output/8PBV/AF \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --precomputed_features_path=/home/koohi/fereidoon/ParallelFold/output/8PBV/AF/feature.pkl \
  --use_gpu_relax=true

gpt-colab
python parjax_gp_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/input/mono_set1/6UF2.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/output/6UF2/colab \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --precomputed_features_path=/home/koohi/fereidoon/ParallelFold/output/6UF2/colab/feature.pkl \
  --use_gpu_relax=true

parjax_gp_gpu0
####-------------------------------------------------------------------------------------------
claude-af
python parjax_cla.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/input/mono_set1/8PBV.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/claude_output/8PBV/AF \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_file=/home/koohi/fereidoon/ParallelFold/output/8PBV/AF/feature.pkl \
  --use_gpu_relax=true


claud-colab
python parjax_cla_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/input/mono_set1/8PBV.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/claude_output/8PBV/colab \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_file=/home/koohi/fereidoon/ParallelFold/output/8PBV/colab/feature.pkl \
  --use_gpu_relax=true


####-------------------------------------------------------------------------------------------
target2

gpt-colab

python parjax_gp.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/target_o_j/input/8UYS/8UYS.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/2LHC \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --precomputed_features_path=/home/koohi/fereidoon/ParallelFold/target_o_j/input/8UYS/colab-feature/feature.pkl \
  --use_gpu_relax=true

gpt-colab

python parjax_gp_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/target_o_j/input/6ZYC/6ZYC.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/target_o_j/6ZYC/123/parjax_gp/colab \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --precomputed_features_path=/home/koohi/fereidoon/ParallelFold/target_o_j/input/6ZYC/AF-feature/feature.pkl \
  --use_gpu_relax=true


8D27
8UYS
####-------------------------------------------------------------------------------------------
target2

claude-af

python parjax_cla.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/2LHC/2LHC.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/2LHC/output \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_file=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/2LHC/feature.pkl \
  --use_gpu_relax=true

claude-colab

python parjax_cla_gpu0.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/8D27/8D27.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/8D27/output \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --feature_file=/home/koohi/fereidoon/ParallelFold/t_o_j_AF_CLA/8D27/feature.pkl \
  --use_gpu_relax=true





####-------------------------------------------------------------------------------------------
google_colab:

claude-af
python parjax_cla.py \
  --fasta_paths=/content/input/pdb_id/pdb_id.fasta \
  --output_dir=/content/output \
  --parameter_path=/content/ParallelFold/alldata/params \
  --precomputed_features_path=/content/ParallelFold/output/GA98/feature.pkl \
  --use_gpu_relax=true


claude-colab
python parjax_cla.py \
  --fasta_paths=/content/ParallelFold/input/mono_set1/GA98.fasta \
  --output_dir=/content/ParallelFold/output \
  --parameter_path=/content/ParallelFold/alldata/params \
  --precomputed_features_path=/content/ParallelFold/output/GA98/feature.pkl \
  --use_gpu_relax=true






















####-------------------------------------------------------------------------------------------
data_dir === -d /home/koohi/fereidoon/ParallelFold/alldata


./run_alphafold.sh \
-d /home/koohi/fereidoon/ParallelFold/alldata \
-o /home/koohi/fereidoon/ParallelFold/output \
-p monomer_ptm \
-i /home/koohi/fereidoon/ParallelFold/input/mono_set1/GA98.fasta \
-m model_1,model_2,model_3,model_4,model_5 \
-t 1800-01-01 \
-c reduced_dbs \
-s \
-P \
-q


### direct run 
fasta_paths === -i .sh

python run_alphafold.py \
  --fasta_paths=/home/koohi/fereidoon/ParallelFold/input/mono_set1/GA98.fasta \
  --output_dir=/home/koohi/fereidoon/ParallelFold/output \
  --parameter_path=/home/koohi/fereidoon/ParallelFold/alldata/params \
  --uniref90_database_path=/home/koohi/fereidoon/ParallelFold/alldata/uniref90/uniref90.fasta \
  --mgnify_database_path=home/koohi/fereidoon/ParallelFold/alldata/mgnify/mgy_clusters_2018_12.fa \
  --template_mmcif_dir=/home/koohi/fereidoon/ParallelFold/alldata/pdb_mmcif/mmcif_files \
  --max_template_date=2020-05-14 \
  --obsolete_pdbs_path=/home/koohi/fereidoon/ParallelFold/alldata/pdb_mmcif/obsolete.dat \
  --use_gpu_relax=true \
  --model_preset=monomer \
  --db_preset=reduced_dbs \
  --small_bfd_database_path=/home/koohi/fereidoon/ParallelFold/alldata/small_bfd/bfd-first_non_consensus_sequences.fasta \
  --pdb70_database_path=/home/koohi/fereidoon/ParallelFold/alldata/pdb70/pdb70 \
  --template_mmcif_dir=/home/koohi/fereidoon/ParallelFold/alldata/pdb_mmcif/mmcif_files \
  --precomputed_features_path=/home/koohi/fereidoon/ParallelFold/output/GA98/feature.pkll 





/content/jaxalpha/org_pdb_AF/6UF2/Cor_str.pdb
/content/jaxalpha/pred_pdb_mmseq_gp/ranked_0/6UF2/query.pdb

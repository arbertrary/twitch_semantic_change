#!/bin/bash

date

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg0/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/results/cosine_metrics.tsv 
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg0/results/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg0/time_series_analysis_standardized_output_f2019-05_l2020-04_alast_clast_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/results/cosine_metrics.tsv 
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg0/results/time_series_analysis_standardized_output_f2019-05_l2020-04_alast_clast_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg0/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/results/cosine_metrics.tsv
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg0/results/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg0/time_series_analysis_standardized_output_f201905_l202004_alast_clast_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg0_lc0_clean0_w2v1/results/cosine_metrics.tsv
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg0/results/time_series_analysis_standardized_output_f201905_l202004_alast_clast_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg1/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/results/cosine_metrics.tsv
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg1/results/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg1/time_series_analysis_standardized_output_f2019-05_l2020-04_alast_clast_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/results/cosine_metrics.tsv
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/ungrouped_results/changepoint/sg1/results/time_series_analysis_standardized_output_f2019-05_l2020-04_alast_clast_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg1/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/results/cosine_metrics.tsv
python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg1/results/time_series_analysis_standardized_output_f2019-05_l2020-04_afirst_cfirst_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

###

printf "\nNew Setting:\n"
python3 compare_pseudowords.py --file1=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/cosine.tsv --column1=1 --file2=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg1/time_series_analysis_standardized_output_f201905_l202004_alast_clast_mcosine_k5_s1000_p0.05_g0_v50.tsv --column2=0 --up_to=50

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/hamilton/vec_128_w5_mc100_iter10_sg1_lc0_clean0_w2v1/results/cosine_metrics.tsv

python3 get_ap_at_50.py --in_file=/home/stud/bernstetter/models/synthetic_twitch/30s_grouped_results/changepoint/sg1/results/time_series_analysis_standardized_output_f201905_l202004_alast_clast_mcosine_k5_s1000_p0.05_g0_v50_metrics.tsv

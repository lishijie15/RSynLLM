# to fill in the following path to evaluation!
output_model=./checkpoints/MoE_Encoder_Heat_GCN_13b_loss_final_
datapath=./power_heat/test_heat/test_heat.json
st_data_path=./power_heat/test_heat/test_heat.pkl
res_path=./result_test/MoE_Encoder_Heat_GCN_13b_loss_final_eval_
start_id=0
end_id=790944
num_gpus=8

python ./RSynLLM/eval/test_RSynLLM.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
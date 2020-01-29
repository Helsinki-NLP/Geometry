

module purge

source /projappl/project_2000945/.bashrc
export PYTHONUSERBASE=/projappl/project_2000945
export PATH=/projappl/project_2000945/bin:$PATH
conda activate senteval


python /projappl/project_2001970/scripts/save_MTembedds.py \
    --model /scratch/project_2001970/AleModel/tr.6l.8ah.en-de_step_200000.pt \
    --only_save #--cuda 


srun -p small --time=2:30:00 --mem=32g --account=project_2001970 python /projappl/project_2001970/scripts/save_MTembedds.py     --model /scratch/project_2001970/AleModel/tr.6l.8ah.en-de_step_200000.pt     --only_save

# CHANGE, to print representations:
/scratch/project_2001970/AleModel/OpenNMT-py/onmt/encoders/transformer.py

torch.save(out,"/users/vazquezc/embeddings.pt")
torch.save(input_norm,"/users/vazquezc/embeddings_norm.pt")

torch.save(out,"/users/vazquezc/layer1.pt")
torch.save(input_norm,"/users/vazquezc/layer1_norm.pt")

torch.save(out,"/users/vazquezc/layer2.pt")
torch.save(input_norm,"/users/vazquezc/layer2_norm.pt")

torch.save(out,"/users/vazquezc/layer3.pt")
torch.save(input_norm,"/users/vazquezc/layer3_norm.pt")
torch.save(out,"/users/vazquezc/layer4.pt")
torch.save(input_norm,"/users/vazquezc/layer4_norm.pt")
torch.save(out,"/users/vazquezc/layer5.pt")
torch.save(input_norm,"/users/vazquezc/layer5_norm.pt")

torch.save(out,"/users/vazquezc/layer6.pt")
torch.save(out,"/users/vazquezc/layer6_norm.pt")

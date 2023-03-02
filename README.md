# T-754-SPLP-Spoken-Language-Processing

## Project Description
The goal of this project is for you to gain hands-on experience with working with a new type of ASR model. That is a large pre-trained ASR model that can be finetuned for specific downstream tasks. Finetuning ASR models is a new paradigm that can potentially have a tremendous effect on the field, especially multilingual models. By leveraging the already-learned knowledge the model has gained during initial training the amount of data needed to create a decent model on unseen data is drastically decreased. With finetuning, these models can be adapted to the specific characteristics of a low-resourced language, leading to improved accuracy and efficiency. 

For this assignment, we will work with the recently published model from OpenAi called Whisper. In this repository, you will find a notebook that has a follow-along guide that will aid you in completing this assignment. In this assignment, you will evaluate the base-model and finetune it to a dataset of your choosing. There are some limitations on what datasets can be used and I will go over that in our first lecture. The project can be summarized as follows:
* Evaluate the base model on Fleurs (https://huggingface.co/datasets/google/fleurs).
* Finetune the model on a dataset of your choosing (discussed in class).
* Have a stab at messing with your data and finetune the base model again. This could be  using a data augmentation method of your choosing on your dataset or by skewing your dataset.
* Evaluate the fine-tuned models on the test portion of your dataset and on Fleurs.

Hand in a report, around 1000-2000 words, where you describe the data you used, the training setup and the results from the finetuning along with a discussion section. Describe the data augmentor and preferably hand in a visualisation of how the data augmentor works.  
* Were the results as expected? 
* Have you come across any limitations to using Whisper?


We will also read the Whisper paper (https://arxiv.org/abs/2212.04356) and hand in a short report with answers and considerations around the following questions:
* How did they curate the training data and how does it differ from conventional ASR models?
* How does this model differ from conventional ASR models?
* Is there anything about the paper that you don’t understand or is unclear?
* Hand in two questions or discussion points from the paper.
* Was the paper useful for you? How so? If not, why not?


The deadline for this will be at 19:00 on 22nd of February, as I will compile discussion points from these answers for the class on the 23rd. 


# Getting started
Start by cloning the repository and run the `01-HuggingFaceDatasets.ipynb` notebook. All of the requirements are in the notebook, installed via pip. Do not run this notebook on the cluster as it will download data which we want to avoid, instead run it locally or on a Google Colab instance. 


# Setup for finetuning
We will use Anaconda to manage our environment. If you do not have Anaconda installed, please install it from https://www.anaconda.com/products/individual. It will pre installed on Terra and the base enviroment should have all the dependencies installed. But if you would like to modify them or setup on you own machine, please follow the instructions below.

## Create a new environment
```
conda create -n whisper python=3.10
```

To activate this environment, use
`conda activate whisper`.
To deactivate an active environment, use
`conda deactivate`.

Then install the following requirements:
```
conda install -c conda-forge jupyter_server=1.23.5
conda install notebook ipykernel jupyterlab 
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

This could take a while (20 minutes ish).


# Terra 
Terra is a shared computation cluster that has a few GTX GPU's. All have around 12GB of memory. Terra has Ubuntu set up and is accessible via SSH while on the RU network. 

## How to use Terra
We use Slurm to schedule jobs on Terra. Slurm is a job scheduler that allows you to submit jobs to the cluster. It will then schedule your job on a node and run it. It controls access too the CPU's, GPU's and allocates memory. If you run a GPU job without Slurm other people's jobs that are running will crash. If you accidently start a GPU job without Slurm please notify the group so that they reschedule their jobs if needed. 


## Communicating with SLURM
Once we have script that we want to run on the cluster we need to communicate with Slurm. We do this by using the `sbatch` command. This command will submit a job to the cluster. The job will be run in the directory where the command is run and return an output log file. The default output file is `slurm-<jobid>.out`. 

To veiw the status of your jobs you can use the `squeue` command. This will show you all the jobs that are currently running on the cluster.

To cancel a job on the cluster you can use the `scancel` command followed by the `job id`. This will cancel the job with the specified job id.

We can specify a few things when submitting a job to Slurm. The most important ones are:
* `--gres=gpu:1` - This will allocate a GPU to the job. 
* `--mem=16G` - This will allocate 16GB of memory to the job.
* `--cpu-per-task=4` - This will allocate 4 CPU's to the job.
* `--output=HelloWorld.out` - This will override the defualt log file to `HelloWorld.out`.

The settings are written in to the sbtach file followed by the command/s. They end with the `.sbtach` extension. An example sbtach file is shown below. 

```
#!/bin/bash
#SBATCH --gres=gpu:1   
#SBATCH --mem=16G
#SBATCH --cpu-per-task=4
#SBATCH --output=HelloWorld.out

echo "Hello World"
```

## Running the training script
The training script both `02-FinetuningWhisper.ipynb` and `04-FinetuningWhisper.py` follow the same flow. We first start by loading the data, prepare and configure the dataloader along with the tokenzier and then setup `Seq2SeqTrainer`. There are a few arguments that we need to configure for our setup. These arguments control how much data we will go through during the fine-tuning. In `run_whisper_training.sbatch` there are a few parameters, not all of them are important for our application but keep these in mind.

```
model_name_or_path          # Path to the model that we will fine-tune.
dataset_name                # Path to the training dataset.
text_column_name            # Name of the column in the dataset where the text is stored. 
language                    # The language we intent do train on, form the avalible languages in Whisper.
train_split_name            # Dictornary key for the training portion of the dataset.
model_index_name            # A descriptive name that you can choose.
max_steps                   # How many fine-tuning steps. Increasing this will increase the length of the training. 
output_dir                  # A path to where we save our fine-tuned model.
per_device_train_batch_size # The number of examples we load to the GPU during training. If we set this to high we will get the a out-of-memmory error but keeping this parameter as high as possible is more efficient during training. The value is usually set as a multible of 2x.   
gradient_accumulation_steps # This parameter helps us to keep efficency in training on `small` GPU's. A rule of thump here is to set `per_device_train_batch_size` * `gradient_accumulation_steps` = 64.
learning_rate               # Initial learning rate. 
warmup_steps                # The number of steps where we increase the learning rate before using linear learning rate decay. 
save_steps                  # At what step interval should we save the model. 
eval_dataset_name           # Path to the evaluation dataset.
eval_split_name             # Dictornary key for the eval portion of the dataset.
eval_steps                  # At what step interval should we evaluate the model.
do_normalize_eval           # Do we want to normalize the text before evaluation.
per_device_eval_batch_size  # The number of examples we load to the GPU.
max_eval_samples            # How many examples do we want to go over during the evaluation.
report_to                   # We will be using Tensorboard here. Which is a very nice tool to visualize the training progress
metric_for_best_model       # WER or CER. 
```

You can read more about the training arguments (here)[https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.Seq2SeqTrainer]. 


Tips and tricks:
When testing the model or some parameters. It is good to change certain parameters so that the model will run through all the steps quicker. For example, setting `max_steps` to 100 will result in a quick run through all the steps. So will setting `per_device_train_batch_size` to 8 or lower and `gradient_accumulation_steps` to 1 will speed up the training process as we will procecss less data. It can also be good to set `eval_steps` and `save_steps` to something low while testing. 




## Monitoring the cluser
To monitor the cluster you can use the `htop` command. This will show you the CPU usage, memory usage and GPU usage. 

To monitor the GPU usage you can use the `nvidia-smi` command. This will show you the GPU usage and memory usage.

I have added an alias for `watch -n0.1` to Terra called wa which can be prepended to any command. Example `wa nvidia-smi` or `wa squeue`. 

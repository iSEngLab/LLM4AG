# EASE Replication Package

This repository contains the code needed to reproduce the experiments.

## Environment Setting up

Please run the following commands to install the necessary packages for running the replication package.

```
conda env create --name ease python=3.9
conda activate ease
pip install pipreqs
pipreqs /the/root/entry --encoding=utf-8 --force
pip install -r requirements.txt 
```

## Folder Structure

```
├── D1
│   ├── CodeBERT
│   ├── CodeGPTori
│   ├── CodeT5
│   ├── GraphCodeBERT
│   └── UniXcoder
├── D2
├── D3
├── README.md
├── RQ1
│   ├── CodeBERT
│   ├── CodeGPTori
│   ├── CodeT5
│   ├── GraphCodeBERT
│   └── UniXcoder
├── RQ2
│   ├── common
│   ├── count.py
│   ├── data
│   │   ├── evo_vocab.npy
│   │   ├── metadata
│   │   └── test_harness
│   ├── eval
│   ├── extractor
│   ├── lib
│   ├── model
│   │   ├── assertion_data.py
│   │   ├── CodeBERT
│   │   ├── CodeGPT
│   │   ├── CodeT5
│   │   ├── GraphCodeBERT
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── UnixCoder
│   ├── naive_ori.py
│   ├── naive.py
│   ├── README.md
│   ├── requirements.txt
│   ├── rq0.sh
│   ├── rqs
│   ├── run_exp_rq3.sh
│   ├── run_exp.sh
│   ├── run_rq1_2.sh
│   ├── run_rq3.sh
│   ├── runscript.py
│   └── toga.py
└── RQ3
    ├── CodeBERT
    ├── CodeGPTori
    ├── CodeT5
    ├── GraphCodeBERT
    ├── IR
    └── UniXcoder

```

## Fine-tuning Model

First, please refer to this [link](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/201250070_smail_nju_edu_cn/EgULVFPJSbFHupUWV53vcnMBLXVKguntHZdxsDM2RQrnrg?e=Jstnas) to download the data.

We have organized the code into corresponding folders based on the research questions in the paper. 

### RQ1

In this RQ, we conducted an evaluation of LLMs' performance in terms of assertion inference accuracy. We used two dataset, namely *Data_new* and *Data_old*, which can be found in *RQ1* folder in the link we provided above. To run the fine-tuning script, you need to run the following commands.

```
cd /the entry of the model you want to run
python runrq1.py
```

The file *runrq1.py* contains two python command, which are the training and inferencing commands respectively.

### RQ2

In this RQ, we evaluated LLMs' performance of fixing real-world bugs on Defects4J benchmark. Based on the replication package provided by *TOGA: A Neural Method for Test Oracle Generation*, we made a few changes according to our experiment. To set up the essential environment for RQ2, please follow the steps below:

#### Preparing the environment

```bash
conda activate ease
pip3 install -r requirements.txt
pip3 uninstall protobuf
pip3 install protobuf==3.20
```

#### Installing Defects4J

- See: https://github.com/rjust/defects4j
- Install v2.0.0
- Patch Defects4j to fix compilation errors

```
diff --git a/framework/projects/defects4j.build.xml b/framework/projects/defects4j.build.xml
index f7065dfc..4e2efdcb 100644
--- a/framework/projects/defects4j.build.xml
+++ b/framework/projects/defects4j.build.xml
@@ -270,6 +270,7 @@ project-specific build file ("project_id"/"project_id".build.xml) for the
                     <!-- Add dependencies to runtime libraries of test generation tools -->
                     <path refid="d4j.lib.testgen.rt"/>
                </classpath>
+               <compilerarg line="-Xmaxerrs 1000"/>
         </javac>
     </target>
```

#### Generating test prefixs and assertions

Please run the following command. Remember to refer to each bash file and modify the argument *MODELNAME* to the model name you need. 

```
bash rq0.sh 
bash run_rq1_2.sh
python -m rqs.rq1_2 MODELNAME
```

### RQ3

In this RQ, we introduced a tool named EASE to enhance LLMs' assertion generation performance. In the folder *IR*, we implemented the IR algorithm used for assertion-retrieval in *IR.py* and the data conjunction method in *data.py*. First, *run IR.py*, it should generate a json file that contains the IR results. Then, run the *data.py* to transform the IR result into the data format that EASE needs. Or you can directly use the dataset we processed *RQ3* folder in the link above. The procedure later is similar to those in RQ1.

```
cd /the entry of the model you want to run
python runrq3.py
```

### D1

In this discussion, we discussed the relationship between inference accuracy and beam size. We use the same dataset as those in RQ1. The commands are as follows:

```
cd /the entry of the model you want to run
python runbeam.py
```

### D2&D3

In these two discussions, we discussed the relationship between inference accuracy and assertion type in D2 and the relationship between inference accuracy and focal-test&assertion length in D3. In these two folders, we provided the data processing script.

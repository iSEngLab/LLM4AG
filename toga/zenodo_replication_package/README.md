# README

## Prepare the environment
```bash
conda create -n revisit_toga python=3.9
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip3 install -r requirements.txt
pip3 uninstall protobuf
pip3 install protobuf==3.20
```

## Install Defects4J
- See: https://github.com/rjust/defects4j
- Install v2.0.0
- Patch Defects4j to fix compilation errors

```diff
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

## Prepare Test Prefixes
- clone TOGA's repository and TOGA's models, see https://github.com/microsoft/toga
- place TOGA's exception model and assertion model to model/exceptions/pretrained/pytorch_model.bin and model/assertions/pretrained/pytorch_model.bin, respectively

```bash
# run EvoSuite 10 times on the buggy and the bug-fixed versions
# generate oracles using TOGA and NoException
# It takes about 20 hours when we using 64 CPU cores
bsah rq0.sh

# run experiments for rq1 and rq2 
bash run_rq1_2.sh
# obtain the results
python -m rqs.rq1_2 CodeT5

# run experiments for rq3
bash run_rq3.sh
# obtain the results
python -m rqs.rq3 cal_result
```
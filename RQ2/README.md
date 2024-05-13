# README

## Prepare the environment
```bash
conda activate ease
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
```bash
bsah rq0.sh

bash run_rq1_2.sh
# obtain the results
python -m rqs.rq1_2 {Model Name}

# run experiments for rq3
bash run_rq3.sh
# obtain the results
python -m rqs.rq3 cal_result
```
package {TEST_PACKAGE};

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
{TEST_IMPORTS}
import java.util.Random;
import org.junit.Assert;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class {TEST_CLASS_NAME}_ESTest extends {TEST_CLASS_NAME}_ESTest_scaffolding {
  {TEST_CASES}
}


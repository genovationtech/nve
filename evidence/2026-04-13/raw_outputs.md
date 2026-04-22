# NVE Benchmark — Full Raw Model Outputs

**Source:** `evidence/experiments/rigorous_comparison.json`  
**Date:** 2026-04-13  
**Platform:** Modal cloud (Intel x86-64, 8 vCPU, 32 GB RAM)  

Every generated text string from every model, scenario, config, and prompt — nothing summarized or truncated.

---

## NVE Framework

Five prompts per run (general relativity, US government branches, photosynthesis, fibonacci, gradient descent).  
Accuracy measured by keyword match in generated text.

### llama1b / unconstrained

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 88% | 3.22 |
| A_quant_only | 88% | 3.65 |
| B_profiled_hot | 88% | 3.97 |
| C_profiled_quant | 0% | 3.74 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the universe is curved by the presence of matter and energy. The theory of general relativity is a theory of gravitation. It is a theory of gravitation that is based on the general theory of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial. The legislative branch is responsible for making laws, the executive branch is responsible for implementing laws, and the judicial branch is responsible for interpreting laws. The legislative branch is
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other organisms use sunlight to convert carbon dioxide and water into food and oxygen. The process is called photosynthesis because it is the process of photosynthesis that produces the light energy that is used to make
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number. 
    >>> fibonacci(5)
    5
    >>> fibonacci(10)
    55
    >>> fibonacci(20)
    6765
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
minimize the cost function. The cost function is a function that measures the difference between the actual output and the desired output. The gradient descent algorithm is used to find the minimum of the cost function. The
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the universe is curved, and that the curvature is caused by the presence of matter. The theory of general relativity is a theory of physics, and is not a theory of biology. The theory of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is responsible for making and passing laws. The executive branch is responsible for carrying out the laws passed by the legislature. The judicial branch is responsible
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
, algae, and some bacteria use sunlight to produce food. It is a vital part of the food chain, and it is the only known process that can convert carbon dioxide into food. Photosynthesis is
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number using a list comprehension and a while loop. 
    The list comprehension should be used to generate the list of numbers, 
    and the while loop should be
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
optimize the parameters of a machine learning model. Gradient descent is a simple and effective way to optimize a model. It is a simple and effective way to optimize a model. Gradient descent is a simple and
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the universe is curved by the presence of matter and energy. The theory of general relativity is a theory of gravitation. It is a theory of gravitation that is based on the general theory of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial. The legislative branch is responsible for making laws, the executive branch is responsible for implementing laws, and the judicial branch is responsible for interpreting laws. The legislative branch is
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other organisms use sunlight to convert carbon dioxide and water into food and oxygen. The process is called photosynthesis because it is the process of photosynthesis that produces the light energy that is used to make
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number. 
    >>> fibonacci(5)
    5
    >>> fibonacci(10)
    55
    >>> fibonacci(20)
    6765
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
minimize the cost function. The cost function is a function that measures the difference between the actual output and the desired output. The gradient descent algorithm is used to find the minimum of the cost function. The
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the following is the case.
The following is the case.
The following is the case.
The following is the case.
The following is the case.
The following is the case.
The following is the
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the following.
1. The first branch of the first branch of the first branch of the first branch of the first branch of the first branch of the first branch of the first branch of the first branch
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
return fibonacci(n) if fibonacci(n) is fibonacci(n) else fibonacci(n) if fibonacci(n) is fibonacci(n) else fibonacci(n) if fibonacci(n) is fibonacci(n) else fibonacci(n)
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the data. The data is the data. The data is the data. The data is the data. The data is the data. The data is the data. The data is the data.
```

---

### llama1b / constrained_2gb

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 88% | 4.23 |
| A_quant_only | 88% | 3.64 |
| B_profiled_hot | 88% | 4.90 |
| C_profiled_quant | 0% | 4.22 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the universe is curved by the presence of matter and energy. The theory of general relativity is a theory of gravitation. It is a theory of gravitation that is based on the general theory of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial. The legislative branch is responsible for making laws, the executive branch is responsible for implementing laws, and the judicial branch is responsible for interpreting laws. The legislative branch is
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other organisms use sunlight to convert carbon dioxide and water into food and oxygen. The process is called photosynthesis because it is the process of photosynthesis that produces the light energy that is used to make
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number. 
    >>> fibonacci(5)
    5
    >>> fibonacci(10)
    55
    >>> fibonacci(20)
    6765
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
minimize the cost function. The cost function is a function that measures the difference between the actual output and the desired output. The gradient descent algorithm is used to find the minimum of the cost function. The
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the universe is curved, and that the curvature is caused by the presence of matter. The theory of general relativity is a theory of physics, and is not a theory of biology. The theory of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is responsible for making and passing laws. The executive branch is responsible for carrying out the laws passed by the legislature. The judicial branch is responsible
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
, algae, and some bacteria use sunlight to produce food. It is a vital part of the food chain, and it is the only known process that can convert carbon dioxide into food. Photosynthesis is
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number using a list comprehension and a while loop. 
    The list comprehension should be used to generate the list of numbers, 
    and the while loop should be
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
optimize the parameters of a machine learning model. Gradient descent is a simple and effective way to optimize a model. It is a simple and effective way to optimize a model. Gradient descent is a simple and
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the universe is curved by the presence of matter and energy. The theory of general relativity is a theory of gravitation. It is a theory of gravitation that is based on the general theory of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial. The legislative branch is responsible for making laws, the executive branch is responsible for implementing laws, and the judicial branch is responsible for interpreting laws. The legislative branch is
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other organisms use sunlight to convert carbon dioxide and water into food and oxygen. The process is called photosynthesis because it is the process of photosynthesis that produces the light energy that is used to make
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number. 
    >>> fibonacci(5)
    5
    >>> fibonacci(10)
    55
    >>> fibonacci(20)
    6765
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
minimize the cost function. The cost function is a function that measures the difference between the actual output and the desired output. The gradient descent algorithm is used to find the minimum of the cost function. The
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the following is the case.
The following is the case.
The following is the case.
The following is the case.
The following is the case.
The following is the case.
The following is the
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the following.
1. The first branch of the first branch of the first branch of the first branch of the first branch of the first branch of the first branch of the first branch of the first branch
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants are grown. The plants
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
return fibonacci(n) if fibonacci(n) is fibonacci(n) else fibonacci(n) if fibonacci(n) is fibonacci(n) else fibonacci(n) if fibonacci(n) is fibonacci(n) else fibonacci(n)
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the data. The data is the data. The data is the data. The data is the data. The data is the data. The data is the data. The data is the data.
```

---

### llama3b / unconstrained

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 100% | 2.12 |
| A_quant_only | 100% | 1.64 |
| B_profiled_hot | 100% | 2.61 |
| C_profiled_quant | 100% | 2.10 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is a result of the curvature of space-time. The curvature of space-time is caused by the presence of mass. The more mass there is, the more the space-time is curved. This is
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is headed by the president, who is elected by the people. The legislative branch is made up of the House of Representatives and the Senate.
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
convert light energy into chemical energy. This energy is stored in the form of glucose, which is then used by the plant to grow and reproduce. Photosynthesis is a complex process that involves a number of
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n <= 1: 
        return n
    else: 
        return fibonacci(n-1) + fibonacci(n-2)
<|end_of_text|>
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a very powerful tool that can be used to solve many problems in machine learning. In this article, we will discuss how to use gradient descent to solve a
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is a result of the curvature of space-time. The curvature of space-time is caused by the presence of mass. The curvature of space-time is measured by the metric tensor. The metric tensor is
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is the Congress, which is made up of the House of Representatives and the Senate. The executive branch is the President and his cabinet. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
convert light energy into chemical energy. It is the basis of the food chain and is essential for life on Earth. Photosynthesis is a complex process that involves many different steps and reactions. The first step
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n <= 1: 
        return n
    else: 
        return fibonacci(n-1) + fibonacci(n-2)
<|end_of_text|>
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
minimize a function. The function is called the loss function. The loss function is a function that measures how far the model is from the data. The goal is to minimize the loss function. The gradient
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is a result of the curvature of space-time. The curvature of space-time is caused by the presence of mass. The more mass there is, the more the space-time is curved. This is
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is headed by the president, who is elected by the people. The legislative branch is made up of the House of Representatives and the Senate.
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
convert light energy into chemical energy. This energy is stored in the form of glucose, which is then used by the plant to grow and reproduce. Photosynthesis is a complex process that involves a number of
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n <= 1: 
        return n
    else: 
        return fibonacci(n-1) + fibonacci(n-2)
<|end_of_text|>
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a very powerful tool that can be used to solve many problems in machine learning. In this article, we will discuss how to use gradient descent to solve a
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is a result of the curvature of space-time. The curvature of space-time is caused by the presence of mass. The more mass there is, the more space-time is curved. The more space
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, the executive, and the judicial. The legislative branch is made up of the House of Representatives and the Senate. The executive branch is made up of the president and the cabinet. The judicial
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other organisms convert light energy into chemical energy. This energy is stored in the bonds of organic molecules, which can then be used to build other molecules or to fuel the growth of the organism. Photos
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n <= 1: 
        return n
    else: 
        return fibonacci(n-1) + fibonacci(n-2) 
<|end_of_text|>
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. The function is usually a loss function, which is a measure of how far the current model is from the desired output. The gradient descent algorithm works by taking small steps
```

---

### llama3b / constrained_2gb

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 100% | 2.31 |
| A_quant_only | 100% | 1.65 |
| B_profiled_hot | 0% | 5.90 |
| C_profiled_quant | 0% | 4.06 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is a result of the curvature of space-time. The curvature of space-time is caused by the presence of mass. The more mass there is, the more the space-time is curved. This is
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is headed by the president, who is elected by the people. The legislative branch is made up of the House of Representatives and the Senate.
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
convert light energy into chemical energy. This energy is stored in the form of glucose, which is then used by the plant to grow and reproduce. Photosynthesis is a complex process that involves a number of
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n <= 1: 
        return n
    else: 
        return fibonacci(n-1) + fibonacci(n-2)
<|end_of_text|>
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a very powerful tool that can be used to solve many problems in machine learning. In this article, we will discuss how to use gradient descent to solve a
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is a result of the curvature of space-time. The curvature of space-time is caused by the presence of mass. The curvature of space-time is measured by the metric tensor. The metric tensor is
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is the Congress, which is made up of the House of Representatives and the Senate. The executive branch is the President and his cabinet. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
convert light energy into chemical energy. It is the basis of the food chain and is essential for life on Earth. Photosynthesis is a complex process that involves many different steps and reactions. The first step
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n <= 1: 
        return n
    else: 
        return fibonacci(n-1) + fibonacci(n-2)
<|end_of_text|>
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
minimize a function. The function is called the loss function. The loss function is a function that measures how far the model is from the data. The goal is to minimize the loss function. The gradient
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
of:: –.
, and and out of the the “ the March,, and the’s of of ( 1.11 the four of of of of of of of of of of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
,, of of, of, of, of,,, a Europe, 2. the, to the of of the for the form of the the for to the a in of the a
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
.. H final: 000- 02  / { 001  /  00 in {00  1 to ^{ ^{ 1 to to  1,, and
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
ies_s  Verdize  <*view their not Mary's's very3s the the'salliver32'se of the'sang3 [22 and@-ger's
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
:. 1. ( 00 00 on on : 10  : :   to  ( with  of  1.   1 .
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
on the of - 22 October 1, 6 01 to to of  of in the (2 of ( 2) a ( & 2)) (  & &
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
,,, of of 2,, a black a and 2,, the the 2 2 a a a a a a a a a a a a a a a a a
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
leaves left, a ball for the a’s’s hit of the the new.
 your the5.
view view.
 1, the { 0. { { { { { { { { {
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
0  201  201  [your{1-1}1}/The 1-1 1-1 1) 1 1 of1 self-
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
. : 1 1  the B of 1  step step step step step  step step of of of  1. of the set  1 method ( 1 on on
```

---

### llama8b / unconstrained

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 88% | 0.79 |
| A_quant_only | 88% | 0.70 |
| B_profiled_hot | 88% | 1.03 |
| C_profiled_quant | 100% | 0.84 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but a consequence of the curvature of spacetime. The curvature is caused by the presence of mass and energy. The theory of general relativity is a theory of gravitation
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The executive branch is the president and the cabinet. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
, some bacteria and some protistans use the energy from sunlight to produce glucose from carbon dioxide and water. This glucose can be converted into pyruvate which can be used to synthesize sugars
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a first-order iterative optimization algorithm for finding the minimum of a function. To find the minimum of a function, we need to find the point where the derivative
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but rather the warping of space and time. This warping is caused by the presence of mass and energy. The theory of general relativity also predicts the existence of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is the president and his cabinet. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other photosynthetic organisms use sunlight to convert carbon dioxide and water into oxygen and glucose. Glucose is a simple sugar that is used by plants as an energy source. Photosynthesis is a complex
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Returns the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. The function is called the cost function. The cost function is a function of the parameters of the model. The parameters are the weights and biases of the model. The
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but a consequence of the curvature of spacetime. The curvature is caused by the presence of mass and energy. The theory of general relativity is a theory of gravitation
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The executive branch is the president and the cabinet. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
, some bacteria and some protistans use the energy from sunlight to produce glucose from carbon dioxide and water. This glucose can be converted into pyruvate which can be used to synthesize sugars
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a first-order iterative optimization algorithm for finding the minimum of a function. To find the minimum of a function, we need to find the point where the derivative
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but rather a consequence of the curvature of spacetime caused by the presence of matter and energy. However, despite the success of the theory in explaining a wide range of phenomena
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is the branch of government that is responsible for the day-to-day operations of the country. The legislative branch is the branch of government that is
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other organisms convert light energy into chemical energy that can be used by the organism. Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy that can be used by the
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
a, b = 0, 1
    while a < n: 
        print(a, end=' ')
        a, b = b, a+b
    print()

fib
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a first-order optimization algorithm. To find the minimum of a function, we need to find the point where the derivative of the function is equal to zero.
```

---

### llama8b / constrained_4gb

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 88% | 1.25 |
| A_quant_only | 88% | 0.71 |
| B_profiled_hot | 0% | 3.40 |
| C_profiled_quant | 0% | 2.26 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but a consequence of the curvature of spacetime. The curvature is caused by the presence of mass and energy. The theory of general relativity is a theory of gravitation
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The executive branch is the president and the cabinet. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
, some bacteria and some protistans use the energy from sunlight to produce glucose from carbon dioxide and water. This glucose can be converted into pyruvate which can be used to synthesize sugars
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a first-order iterative optimization algorithm for finding the minimum of a function. To find the minimum of a function, we need to find the point where the derivative
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but rather the warping of space and time. This warping is caused by the presence of mass and energy. The theory of general relativity also predicts the existence of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is the president and his cabinet. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other photosynthetic organisms use sunlight to convert carbon dioxide and water into oxygen and glucose. Glucose is a simple sugar that is used by plants as an energy source. Photosynthesis is a complex
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Returns the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. The function is called the cost function. The cost function is a function of the parameters of the model. The parameters are the weights and biases of the model. The
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
ith uncio.laces.linger.              clmt          · of   Tud   언        igh     Heavenly -))-
       Riley   agger
agger
agger
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
9  Q

  ’s
  artss96
 Arts .


  
     

  
50
 icer50
 due
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
Germanyearned Unitedands would UVandsand 257andsagas Oanda Os79 Vine IS No  Ly  Isands  Valley Is Valley ValleyAND’s VandesAND. Vande  Valleyand
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
erverAIDES.VSidos. besidos andorda,orda orda orda tumоряд,orda Tib古屋. tan. ( SmCit. sm. Beard,orda .yd. Beard,
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
oso 390 good Wonder 390ardon 390 uba  pld
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
of [ p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p p
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
on.kp  at... ...,,,,......,,,,,,,,,,USD ... ...USDUSD,, ...USDUSDUSDUSD
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
1pp000000000000000000000000000000000001010
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
of / of al s s al s s s s s s s s s s s s s s s s s sol tò̀ol s sol...̀̀̀ol...̀
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
a, at,,,,,,,,,,,,,,,,,,,d,0,0,0d p7d7 p7d7 p7
```

---

### llama8b / constrained_8gb

| Config | Accuracy | Tok/s |
|--------|----------|-------|
| baseline | 88% | 0.55 |
| A_quant_only | 88% | 0.87 |
| B_profiled_hot | 12% | 1.13 |
| C_profiled_quant | 0% | 1.37 |

#### Config: baseline

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but a consequence of the curvature of spacetime. The curvature is caused by the presence of mass and energy. The theory of general relativity is a theory of gravitation
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the legislative, executive, and judicial branches. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The executive branch is the president and the cabinet. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
, some bacteria and some protistans use the energy from sunlight to produce glucose from carbon dioxide and water. This glucose can be converted into pyruvate which can be used to synthesize sugars
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Return the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. It is a first-order iterative optimization algorithm for finding the minimum of a function. To find the minimum of a function, we need to find the point where the derivative
```

#### Config: A_quant_only

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
gravity is not a force, but rather the warping of space and time. This warping is caused by the presence of mass and energy. The theory of general relativity also predicts the existence of
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the executive, legislative, and judicial branches. The executive branch is the president and his cabinet. The legislative branch is the Congress, which is made up of the Senate and the House of Representatives. The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and other photosynthetic organisms use sunlight to convert carbon dioxide and water into oxygen and glucose. Glucose is a simple sugar that is used by plants as an energy source. Photosynthesis is a complex
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
"""Returns the nth Fibonacci number."""
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else:
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
find the minimum of a function. The function is called the cost function. The cost function is a function of the parameters of the model. The parameters are the weights and biases of the model. The
```

#### Config: B_profiled_hot

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
you can find more information about the location for you to find a more information about the people who on the Internet, a more information.
<|end_of_text|>
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the most popular for a reason, the most expensive for a time.
 the most.
 the most important changes in the city, the United.
 the most.<|end_of_text|>
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
and the pictures the organization of your way to the university’s description will be the world’s are on a large, very, a a the world’s a very. the University of the world) a
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
, a man, a man in a man in a man in a man in a man in a man in a man in a man in a man in a man in a man in a man
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
get the right number of share from 2, 3 1.
  1 1, 1. 1. 1. 1. 1. 1.
```

#### Config: C_profiled_quant

**Prompt:** `The theory of general relativity explains that`  
**Match:** —
```
the law that is in the definition of the law that is the law of the road the law for the law for the law of the law the the law the the law the the the the the the
```

**Prompt:** `The three branches of the United States government are`  
**Match:** —
```
the most and the most in the,, the the the the the the the the the the the the the the the the the the the the the the the the the the the the The The The
```

**Prompt:** `Photosynthesis is the process by which plants`  
**Match:** —
```
to die on the  1st, the man to be, the 1, the, the, the the the the the the the the the the the the the the the the the the
```

**Prompt:** `def fibonacci(n):`  
**Match:** —
```
you were using in the last year of the time you had, the number of people of the number of people in the number of people to do the number of people to do the number of people to
```

**Prompt:** `In machine learning, gradient descent is used to`  
**Match:** —
```
be of the 1992 way, the 2010 to the  100th  100  the    the    the  of the   the  the the
```

---

## llama.cpp (GGUF Quantized)

Eight prompts per run:
1. `The capital of France is`
2. `Water is composed of hydrogen and`
3. `The largest planet in the solar system is`
4. `If today is Monday, tomorrow is`
5. `A square has four equal sides. A shape with four equal sides and four right angles is a`
6. `def add(a, b): return a`
7. `# Python: list of squares 0-4
squares = [x**2 for x in`
8. `The main benefit of regular exercise is improved`

### llama1b / unconstrained / q4
**Accuracy:** 88% | **Tok/s:** 3.43

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris, and it is the most visited city in the world. It is a city of art, culture, and history. It
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The hydrogen is in the form of protons and the oxygen is in the form of electrons. The elec
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✗
```
the Earth. It is the only planet in the solar system that has water on its surface. It is the only
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day is it on the 3rd day of the week?
The an
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. The area of a square is the number of square units that make up the shape. The area of a sq
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]

# Python: list of cubes 0-4
cubes = [x**3 for x in range(5)]

# Python: list of cubes 0-
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
health and fitness. However, it is also important to remember that exercise is not a cure-all. It i
```

---

### llama1b / unconstrained / q8
**Accuracy:** 100% | **Tok/s:** 12.79

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris. It is the most visited city in the world. It is also the most expensive city in the world. T
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The hydrogen is the lightest element and the oxygen is the heaviest. The hydrogen is the mo
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
the one that is closest to the sun. It is also the one that is the most massive. The planet is call
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day is it on Thursday?
A. Tuesday
B. Wednesd
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. The sides of a square are equal and the angles are right angles. The sides of a square are
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of cubes 0-4
cubes = [x**3 for x in range(5)]
print(cubes)
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
cardiovascular health. It also helps to reduce the risk of developing diabetes, high blood pressure
```

---

### llama1b / constrained_2gb / q4
**Accuracy:** 88% | **Tok/s:** 4.00

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris, and it is the most visited city in the world. It is a city of art, culture, and history. It
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The hydrogen is in the form of protons and the oxygen is in the form of electrons. The elec
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✗
```
the Earth. It is the only planet in the solar system that has water on its surface. It is the only
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day is it on the 3rd day of the week?
The an
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. The area of a square is the number of square units that make up the shape. The area of a sq
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]

# Python: list of cubes 0-4
cubes = [x**3 for x in range(5)]

# Python: list of cubes 0-
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
health and fitness. However, it is also important to remember that exercise is not a cure-all. It i
```

---

### llama1b / constrained_2gb / q8
**Accuracy:** 100% | **Tok/s:** 12.03

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris. It is the most visited city in the world. It is also the most expensive city in the world. T
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The hydrogen is the lightest element and the oxygen is the heaviest. The hydrogen is the mo
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
the one that is closest to the sun. It is also the one that is the most massive. The planet is call
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day is it on Thursday?
A. Tuesday
B. Wednesd
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. The sides of a square are equal and the angles are right angles. The sides of a square are
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of cubes 0-4
cubes = [x**3 for x in range(5)]
print(cubes)
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
cardiovascular health. It also helps to reduce the risk of developing diabetes, high blood pressure
```

---

### llama3b / unconstrained / q4
**Accuracy:** 100% | **Tok/s:** 1.57

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris. It is the largest city in France and the capital of the Île-de-France region. The city of Pa
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The ratio of hydrogen to oxygen is 2:1. The ratio of hydrogen to oxygen is 2:1. The ratio o
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
Jupiter. It is the fifth planet from the Sun and the largest planet in the Solar System. It is a ga
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, then the day after the day after tomorrow is Thur
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. A square has four right angles. A square has four equal sides. A square has four equal side
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of squares 0-4
squares = [x**2 for x in range(5)]
print(sq
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
cardiovascular health. Regular exercise can help to reduce the risk of heart disease, stroke, and o
```

---

### llama3b / unconstrained / q8
**Accuracy:** 100% | **Tok/s:** 3.11

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris. It is the largest city in France and the capital of the Île-de-France region. The city of li
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The ratio of hydrogen to oxygen is 2:1. The ratio of hydrogen to oxygen is 2:1. The ratio o
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
Jupiter. It is a gas giant. Jupiter is the fifth planet from the sun. It is the largest planet in t
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and Wednesday is Thursday, then Thursday is Friday. This is an example of a what?
A. logic
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. A square has four right angles. A square has four equal sides. A square has four equal angl
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of squares 0-4, with condition
squares = [x**2 for x in ra
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
health. Exercise can help you lose weight, lower your blood pressure, and reduce your risk of heart
```

---

### llama3b / constrained_2gb / q4
**Accuracy:** — | **Tok/s:** —

*(OOM — no tokens generated)*

---

### llama3b / constrained_2gb / q8
**Accuracy:** — | **Tok/s:** —

*(OOM — no tokens generated)*

---

### llama8b / unconstrained / q4
**Accuracy:** 100% | **Tok/s:** 0.85

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is Par
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The ratio of hydrogen to oxygen in water is 2:16. The ratio of hydrogen to oxygen in water
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
Jupiter. It is a gas giant, which means that it is made up of mostly hydrogen and helium. Jupiter i
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day of the week is 4 days after tomorrow?
A.
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. A square is a special type of rectangle and a special type of parallelogram. A square is a
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of squares 0-4
squares = [x**2 for x in range(5)]
print(sq
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
physical fitness, which can reduce the risk of developing various health problems. Regular exercise
```

---

### llama8b / unconstrained / q8
**Accuracy:** 100% | **Tok/s:** 2.17

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Par
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. A molecule of water contains two atoms of hydrogen and one atom of oxygen. The chemical for
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
Jupiter. It is a gas giant, meaning that it is made up of mostly hydrogen and helium. Jupiter is th
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day of the week is 4 days after tomorrow?
A.
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. A square is a special case of a rectangle. A square is a special case of a rhombus. A squar
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of squares 0-4
squares = [x**2 for x in range(5)]
print(sq
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
cardiovascular health. Exercise helps to lower blood pressure, improve cholesterol levels, and redu
```

---

### llama8b / constrained_4gb / q4
**Accuracy:** — | **Tok/s:** —

*(OOM — no tokens generated)*

---

### llama8b / constrained_4gb / q8
**Accuracy:** — | **Tok/s:** —

*(OOM — no tokens generated)*

---

### llama8b / constrained_8gb / q4
**Accuracy:** 100% | **Tok/s:** 0.71

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is Par
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The ratio of hydrogen to oxygen in water is 2:16. The ratio of hydrogen to oxygen in water
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
Jupiter. It is a gas giant, which means that it is made up of mostly hydrogen and helium. Jupiter i
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day of the week is 4 days after tomorrow?
A.
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. A square is a special type of rectangle and a special type of parallelogram. A square is a
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of squares 0-4
squares = [x**2 for x in range(5)]
print(sq
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
physical fitness, which can reduce the risk of developing various health problems. Regular exercise
```

---

### llama8b / constrained_8gb / q8
**Accuracy:** — | **Tok/s:** —

*(OOM — no tokens generated)*

---

## HuggingFace fp32

Same eight-prompt suite as llama.cpp.

### llama1b / unconstrained
**Accuracy:** 100% | **Tok/s:** 3.47

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✓
```
Paris. It is the most populous city in France and the country’s capital and largest city. Paris is
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. The hydrogen is the lightest element and the oxygen is the heaviest. The hydrogen is the mo
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
the one that is closest to the sun. It is also the one that is the most massive. The planet is call
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and the day after tomorrow is Wednesday, what day is it on Thursday?
A. Tuesday
B. Wednesd
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. The sides of a square are equal. The sides of a square are equal. The sides of a square are
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of cubes 0-4
cubes = [x**3 for x in range(5)]
print(cubes)
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
cardiovascular health. It also helps to reduce the risk of heart disease, high blood pressure, and
```

---

### llama1b / constrained_2gb
**Accuracy:** — | **Tok/s:** —

*(No outputs — oom_predicted)*

---

### llama8b / unconstrained
**Accuracy:** 88% | **Tok/s:** 0.51

**Prompt:** `The capital of France is`  
**Expected:** `paris` | **Match:** ✗
```
a city of many faces. It is a city of history, culture, and art. It is a city of fashion, food, and
```

**Prompt:** `Water is composed of hydrogen and`  
**Expected:** `oxygen` | **Match:** ✓
```
oxygen. A molecule of water contains one oxygen atom and two hydrogen atoms. The ratio of hydrogen
```

**Prompt:** `The largest planet in the solar system is`  
**Expected:** `jupiter` | **Match:** ✓
```
Jupiter. It is a gas giant, which means that it is made up of mostly hydrogen and helium. Jupiter i
```

**Prompt:** `If today is Monday, tomorrow is`  
**Expected:** `tuesday` | **Match:** ✓
```
Tuesday, and Wednesday is Thursday, what day is Friday?
If today is Monday, tomorrow is Tuesday, an
```

**Prompt:** `A square has four equal sides. A shape with four equal sides and four right angles is a`  
**Expected:** `square` | **Match:** ✓
```
square. A square is a special case of a rhombus, which has all equal sides and equal angles. A squa
```

**Prompt:** `def add(a, b):
    return a`  
**Expected:** `+` | **Match:** ✓
```
+ b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
```

**Prompt:** `# Python: list of squares 0-4
squares = [x**2 for x in`  
**Expected:** `range` | **Match:** ✓
```
range(5)]
print(squares)

# Python: list of squares 0-4
squares = [x**2 for x in range(5)]
print(sq
```

**Prompt:** `The main benefit of regular exercise is improved`  
**Expected:** `health` | **Match:** ✓
```
health. Exercise can help prevent heart disease, stroke, diabetes, obesity, and high blood pressure
```

---

### llama8b / constrained_4gb
**Accuracy:** — | **Tok/s:** —

*(No outputs — oom_predicted)*

---

### llama8b / constrained_8gb
**Accuracy:** — | **Tok/s:** —

*(No outputs — oom_predicted)*

---


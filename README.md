**DeepSeekJanus**  
DeepSeek released an open source multi-modal text to image model, named Janus.  

**A. In this exploration, there are 2 exercises.  **
1. We attempt to reproduce an image from a code shared by Daniel Corin.
2. We attempt to improve the code with new features.

**B. How is it Done?  **  
Preparation of programming environment is neccessary before running the code.
1. Installation of Pytorch, a machine learning language, is needed on top of Python.
2. To download Janus Model, we have to clone Janus Model from Github repository.
3. Janus will be downloaded when we run the code for the first time.

**C. Problems Encountered**  
1. Error encountered was <ModuleNotFoundError: No module named 'janus.models'>
2. The root cause was due to the code file MyJanusPro7B.py is not in the same path as Janus.
3. We moved MyJanusPro7B.py into Janus folder, and 'janus.models' was able to load.

**C. Deliverables**  
1. We are able to reproduce original image of the brown dog from Daniel Corin's code.

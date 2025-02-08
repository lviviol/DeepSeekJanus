**DeepSeekJanus**  
DeepSeek released an open source multi-modal text to image model, named Janus.  

**A. In this exploration, there are 2 exercises.**
1. We attempt to reproduce an image from a code shared by Daniel Corin.
2. We attempt to improve the code with new features.

**B. How is it Done?**  
Preparation of programming environment is neccessary before running the code.
1. Installation of Pytorch, a machine learning language, is needed on top of Python.
2. To download Janus Model, we have to clone Janus Model from Github repository.
3. Janus will be downloaded when we run the code for the first time.

**C. Problems Encountered**  
1. Error encountered was <ModuleNotFoundError: No module named 'janus.models'>
2. The root cause was due to the code file MyJanusPro7B.py is not in the same path as Janus.
3. We moved MyJanusPro7B.py into Janus folder, and 'janus.models' was able to load.

**D. Deliverables**  
1. We are able to reproduce original image of the brown dog from Daniel Corin's code.
   ![alt text](https://github.com/lviviol/DeepSeekJanus/blob/9068ff0db2ddd61a7d5311d17ce9da37e993cd92/img_0.jpg?raw=true)

2. We modifed author's prompt with our own as follows.  
2 person are walking on the beach towards the ocean.
The ocean has 2 shades of color, with baby blue in front and deep blue at the back.
In the mids of the ocean, 3 dolphins are jumping out of the ocean.
The 3 dolphins formed a circle while they are in the air.
The sky is beautiful blue with some clouds.


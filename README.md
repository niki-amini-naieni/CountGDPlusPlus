# CountGD++: Generalized Prompting for Open-World Counting

Niki Amini-Naieni & Andrew Zisserman

Official PyTorch implementation for CountGD++. Details can be found in the paper, [[Paper]]() [[Project page]](https://github.com/niki-amini-naieni/CountGDPlusPlus/).

If you find this repository useful, please give it a star ⭐.

<img src=img/teaser.jpg width="100%"/>
<strong>New capabilities of COUNTGD++.</strong>
<em>(a) Counting with Positive & Negative Prompts:</em> The negative visual exemplar enables CountGD++ to differentiate between cells that have the same round shape as the object to count but are of a different appearance;  
<em>(b) Pseudo-Exemplars:</em> Pseudo-exemplars are automatically detected from text-only input and fed back to the model, improving the accuracy of the final count for objects, like unfamiliar fruits, that are challenging to identify given text alone.

## CountGD++ Architecture
<img src=img/inference-architecture.jpg width="100%"/>

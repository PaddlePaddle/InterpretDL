Abstract Interpreter
===========================

.. autoclass:: interpretdl.Interpreter
   :members:
   :private-members:

Sub-abstract Interpreters
===========================

Input Gradient Interpreter
--------------------------

.. autoclass:: interpretdl.InputGradientInterpreter
   :members:
   :private-members:

Input Output Interpreter
--------------------------

.. autoclass:: interpretdl.InputOutputInterpreter
   :members:
   :private-members:

Intermediate-Layer Interpreter
------------------------------
.. autoclass:: interpretdl.IntermediateLayerInterpreter
   :members:
   :private-members:


Input Feature Interpreters
===========================
   
Smooth Gradients
----------------

.. autoclass:: interpretdl.SmoothGradInterpreter
   :members:

Integrated Gradients
--------------------

.. autoclass:: interpretdl.IntGradCVInterpreter
   :members:

.. autoclass:: interpretdl.IntGradNLPInterpreter
   :members:
   
Occlusion
---------

.. autoclass:: interpretdl.OcclusionInterpreter
   :members:
      
Gradient Shap
-------------

.. autoclass:: interpretdl.GradShapCVInterpreter
   :members:

.. autoclass:: interpretdl.GradShapNLPInterpreter
   :members:
   
LIME
----

.. autoclass:: interpretdl.LIMECVInterpreter
   :members:

.. autoclass:: interpretdl.LIMENLPInterpreter
   :members:

LIME With Global Prior
----------------------

.. autoclass:: interpretdl.LIMEPriorInterpreter
   :members:

NormLIME
--------

.. autoclass:: interpretdl.NormLIMECVInterpreter
   :members:

.. autoclass:: interpretdl.NormLIMENLPInterpreter
  :members:

LRP
----------------

.. autoclass:: interpretdl.LRPCVInterpreter
   :members:
  
Intermediate-Layer Feature Interpreters
=========================================

Grad-CAM
--------

.. autoclass:: interpretdl.GradCAMInterpreter
   :members:

Score CAM
---------

.. autoclass:: interpretdl.ScoreCAMInterpreter
   :members:

Rollout
--------

.. autoclass:: interpretdl.RolloutInterpreter
   :members:

TAM
--------

.. autoclass:: interpretdl.TAMInterpreter
   :members:


Dataset & Training-Process Interpreters
============================================

Forgetting Events
-----------------

.. autoclass:: interpretdl.ForgettingEventsInterpreter
   :members:

SGDNoise
-----------------

See Tutorials.

TrainIng Data analYzer (TIDY)
----------------------------------

See Tutorials.

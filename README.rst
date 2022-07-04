[ICCV 2021- Oral] PyTorch Implementation of `Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers <https://arxiv.org/abs/2103.15679>`_
=================================================================================================================================================================

|youtube|

.. |youtube| image:: https://img.shields.io/static/v1?label=ICCV2021&message=12MinuteVideo&color=red
                   :target: https://www.youtube.com/watch?v=bQTL34Dln-M

Notebooks for LXMERT + DETR:
----------------------------

|DETR_LXMERT|

.. |DETR_LXMERT| image:: https://colab.research.google.com/assets/colab-badge.svg
                   :target: https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_Explainability.ipynb

Notebook for CLIP:
----------------------------

|CLIP|

.. |CLIP| image:: https://colab.research.google.com/assets/colab-badge.svg
                   :target: https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb

**Demo**: You can check out a demo on `Huggingface spaces <https://huggingface.co/spaces/PaulHilders/CLIPGroundingExplainability>`_ or scan the following QR code.

.. image:: https://user-images.githubusercontent.com/19412343/176676771-d26f2146-9901-49e7-99be-b030f3d790de.png
   :width: 100


Notebook for ViT:
----------------------------

|ViT|

.. |ViT| image:: https://colab.research.google.com/assets/colab-badge.svg
                   :target: https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_explainability_ViT.ipynb

.. sectnum::


Using Colab
----------------

* Please notice that the notebook assumes that you are using a GPU. To switch runtime go to Runtime -> change runtime type and select GPU.
* Installing all the requirements may take some time. After installation, please restart the runtime.

Running Examples
----------------

Notice that we have two `jupyter` notebooks to run the examples presented in the paper.

* `The notebook for LXMERT <./LXMERT.ipynb>`_ contains both the examples from the paper and examples with images from the internet and free form questions.
  To use your own input, simply change the `URL` variable to your image and the `question` variable to your free form question.

  .. image:: LXMERT.PNG

  .. image:: LXMERT-web.PNG

* `The notebook for DETR <./DETR.ipynb>`_ contains the examples from the paper.
  To use your own input, simply change the `URL` variable to your image.

  .. image:: DETR.PNG

Reproduction of results
-----------------------

^^^^^^^^^^
VisualBERT
^^^^^^^^^^

Run the `run.py` script as follows:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python VisualBERT/run.py --method=<method_name> --is-text-pert=<true/false> --is-positive-pert=<true/false> --num-samples=10000 config=projects/visual_bert/configs/vqa2/defaults.yaml model=visual_bert dataset=vqa2 run_type=val checkpoint.resume_zoo=visual_bert.finetuned.vqa2.from_coco_train env.data_dir=/path/to/data_dir training.num_workers=0 training.batch_size=1 training.trainer=mmf_pert training.seed=1234

.. note::

  If the datasets aren't already in `env.data_dir`, then the script will download the data automatically to the path in `env.data_dir`.


^^^^^^
LXMERT
^^^^^^

#. Download `valid.json <https://nlp.cs.unc.edu/data/lxmert_data/vqa/valid.json>`_:

    .. code-block:: bash

      pushd data/vqa
      wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/valid.json
      popd

#. Download the `COCO_val2014` set to your local machine.

   .. note::

      If you already downloaded `COCO_val2014` for the `VisualBERT`_ tests, you can simply use the same path you used for `VisualBERT`_.

#. Run the `perturbation.py` script as follows:

    .. code-block:: bash

      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python lxmert/lxmert/perturbation.py  --COCO_path /path/to/COCO_val2014 --method <method_name> --is-text-pert <true/false> --is-positive-pert <true/false>



^^^^
DETR
^^^^

#. Download the COCO dataset as described in the `DETR repository <https://github.com/facebookresearch/detr#data-preparation>`_.
   Notice you only need the validation set.
   
#. Lower the IoU minimum threshold from 0.5 to 0.2 using the following steps:
         
   * Locate the `cocoeval.py` script in your python library path:
      
     find library path:
    
      .. code-block:: python

         import sys
         print(sys.path)
         
     find `cocoeval.py`: 
  
      .. code-block:: bash
      
         cd /path/to/lib
         find -name cocoeval.py
         
   * Change the `self.iouThrs` value in the `setDetParams` function (which sets the parameters for the COCO detection evaluation) in the `Params` class as follows:
      
     insead of:
    
      .. code-block:: python

       self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
     use: 
  
      .. code-block:: python

       self.iouThrs = np.linspace(.2, 0.95, int(np.round((0.95 - .2) / .05)) + 1, endpoint=True)

#. Run the segmentation experiment, use the following command:

    .. code-block:: bash

       CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd`  python DETR/main.py --coco_path /path/to/coco/dataset  --eval --masks --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --batch_size 1 --method <method_name>

Citing
-------

If you make use of our work, please cite our paper:

    .. code-block:: latex

       @InProceedings{Chefer_2021_ICCV,
          author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
          title     = {Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers},
          booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
          month     = {October},
          year      = {2021},
          pages     = {397-406}
       }


Credits
-------

* VisualBERT implementation is based on the `MMF <https://github.com/facebookresearch/mmf>`_ framework.
* LXMERT implementation is based on the `offical LXMERT <https://github.com/airsplay/lxmert>`_ implementation and on `Hugging Face Transformers <https://github.com/huggingface/transformers>`_.
* DETR implementation is based on the `offical DETR <https://github.com/facebookresearch/detr>`_ implementation.
* CLIP implementation is based on the `offical CLIP <https://github.com/openai/CLIP>`_ implementation.
* The CLIP huggingface spaces demo was made by Paul Hilders, Danilo de Goede, and Piyush Bagad from the University of Amsterdam as part of their `final project <https://github.com/bpiyush/CLIP-grounding>`_.

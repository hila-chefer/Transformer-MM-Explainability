Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers
=================================================================================================

.. sectnum::

Running Examples
----------------

Notice that we have two `jupyter` notebooks to run the examples presented in the paper.

* `The notebook for LXMERT <./LXMERT.ipynb>`_ contains both the examples from the paper and examples with images from the internet and free form questions.
  To use your own input, simply change the `URL` variable to your image and the `question` variable to your free form question.

  .. image:: LXMERT.PNG

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

      cd data/vqa
      wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/valid.json

#. Download the `COCO_val2014` set to your local machine.

   .. note::

      if you already downloaded `COCO_val2014` for the VisualBERT tests, you can simply use the same path you used for VisualBERT.

#. Run the `perturbation.py` script as follows:

    .. code-block:: bash

      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python lxmert/lxmert/perturbation.py  --COCO_path /path/to/COCO_val2014 --method <method_name> --is-text-pert <true/false> --is-positive-pert <true/false>



^^^^
DETR
^^^^

#. Download the COCO dataset as described in the `DETR repository <https://github.com/facebookresearch/detr#data-preparation>`_.
   Notice you only need the validation set.

#. Run the segmentation experiment, use the following command:

    .. code-block:: bash

       CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd`  python DETR/main.py --coco_path /path/to/coco/dataset  --eval --masks --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --batch_size 1 --method <method_name>

Credits
-------

* VisualBERT implementation is based on the `MMF <https://github.com/facebookresearch/mmf>`_ framework.
* LXMERT implementation is based on the `offical LXMERT <https://github.com/airsplay/lxmert>`_ implementation and on `Hugging Face Transformers <https://github.com/huggingface/transformers>`_.
* DETR implementation is based on the `offical DETR <https://github.com/facebookresearch/detr>`_ implementation

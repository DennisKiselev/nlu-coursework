# NLU Coursework

The chosen task for this submission was NLI, using methods B and C.

## Code structure

The following describes the structure of the submission. Files end in "_B" or "_C" to indicate which method the file is used for:

- **Test predictions** - Contained in correct format in **Group_17_B.csv** and and **Group_17_C.csv**
- **Training code** - Contained in python notebooks **training_B.ipynb** and **training_C.ipynb**
    - At the top of the notebook, this variable need to be set:
      - *cls_filepath* - path to the **classes** directory, which should contain **evaluation.py** and **preprocessing.py**
- **Evaluation code** - Contained in python notebooks **evaluation_B.ipynb** and **evaluation_C.ipynb**
  - Needs to be placed in the same directory as its model files
  - Outputs metrics in code block, and predictions to a csv file
  - At the top of the notebook, these variables need to be set:
    - *cls_filepath* - path to the **classes** directory, which should contain **evaluation.py** and **preprocessing.py**
    - *dataset_path* - path to dataset used for evaluation, in **.csv** format
    - *labels_path* - path to output predictions to, in **.csv** format
    - For method B:
      - *model_weight_filepath* - path to model weights, in **.hdf5** format
      - *model_arch_filepath* - path to model architecture file, in **.json** format
    - For method C:
      - *model_filepath* - path to model, in **.pt** format
- **Demo code** - Contained in python notebooks **demo_code_B.ipynb** and **demo_code_C.ipynb**
  - Needs to be placed in the same directory as its model files
  - Outputs predictions to a csv file
  - At the top of the notebook, these variables need to be set:
    - *dataset_path* - path to dataset used for evaluation, in **.csv** format
    - *labels_path* - path to output predictions to, in **.csv** format
    - For method B:
      - *model_weight_filepath* - path to model weights, in **.hdf5** format
      - *model_arch_filepath* - path to model architecture file, in **.json** format
    - For method C:
      - *model_filepath* - path to model, in **.pt** format
- **Model cards** - Contained in markdown format in **model_card_B.md** and **model_card_C.md**
- **Poster** - Contained in PDF format in **poster.pdf**

## Data sources

XLNet for embeddings for method B from https://huggingface.co/docs/transformers/en/model_doc/xlnet

RoBERTa for method C from https://huggingface.co/docs/transformers/en/model_doc/roberta

## Links to models

**Method B** - **solution_B.hdf5** and **solution_B.json** at https://drive.google.com/drive/folders/1mco0KXYAJUdsprtzFp0sZC-3mhb-y0nZ?usp=sharing

**Method C** - **solution_C.pt** at https://drive.google.com/drive/folders/1nGpmByFpKSMoiNxrzOT_0XC2q69QwPwO?usp=sharing

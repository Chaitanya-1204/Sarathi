
from logger import log

def create_test_questions():
    """
    Defines a list of 10 tricky and diverse questions to test the RAG agent.
    """
    questions = [
        # Question 1: The Core Architecture: MED
        """
        The README.md mentions that BLIP is a Multimodal Mixture of Encoder-Decoder (MED). 
        Looking at `models/blip.py`, how are the `BLIP_Base` and `BLIP_Decoder` classes 
        structured to reflect this encoder-decoder architecture, and what are the roles 
        of the visual encoder and text encoder/decoder?
        """,

        # Question 2: Pre-training Objectives
        """
        The `pretrain.py` script is central to BLIP's learning process. What are the three 
        pre-training objectives implemented in the `BLIP_Pretrain` class in `models/blip_pretrain.py`, 
        and how do the `loss_ita`, `loss_itm`, and `loss_lm` variables in the `train` function 
        correspond to these objectives?
        """,

        # Question 3: Fine-tuning for Different Tasks
        """
        The codebase includes scripts for fine-tuning on various tasks like Image Captioning 
        (`train_caption.py`), VQA (`train_vqa.py`), and Image-Text Retrieval (`train_retrieval.py`). 
        How do the model architectures in `models/blip_caption.py`, `models/blip_vqa.py`, 
        and `models/blip_retrieval.py` differ from the pre-training model to adapt to these specific tasks?
        """,

        # Question 4: Data Handling and Augmentation
        """
        The `data/__init__.py` script defines the `create_dataset` function, which is used 
        across all training scripts. What are the key data augmentation techniques applied 
        to the training images in `transform_train`, and how do they differ from the 
        transformations in `transform_test`?
        """,

        # Question 5: Configuration and Hyperparameters
        """
        The `configs` directory contains `.yaml` files for each task (e.g., `caption_coco.yaml`, 
        `vqa.yaml`). How are these configuration files used in the training scripts, and what 
        are some of the key hyperparameters that can be tuned for a task like image captioning?
        """,

        # Question 6: Vision Transformer (ViT) Backbone
        """
        The `create_vit` function in `models/blip.py` is used to create the visual encoder. 
        What are the different ViT architectures available (e.g., 'base', 'large'), and how 
        does the code handle loading pre-trained weights for the ViT, as seen in 
        `models/blip_pretrain.py`?
        """,

        # Question 7: Image-Text Matching (ITM) Logic
        """
        In `models/blip_itm.py`, the `BLIP_ITM` class has a forward pass that can be used for 
        either Image-Text Matching ('itm') or Image-Text Contrastive learning ('itc'). What is 
        the difference in the forward pass logic for these two heads, and how are the 
        similarity scores calculated in each case?
        """,

        # Question 8: Visual Question Answering (VQA) specific design
        """
        The `BLIP_VQA` model in `models/blip_vqa.py` uses both a text encoder and a text decoder. 
        How are the question and image embeddings combined and processed by the text encoder 
        before being passed to the text decoder to generate an answer?
        """,

        # Question 9: Zero-shot Video-Text Retrieval
        """
        The `eval_retrieval_video.py` script demonstrates zero-shot video-text retrieval. 
        How does the code adapt the image-based BLIP model to handle video inputs, as seen 
        in the `evaluation` function?
        """,

        # Question 10: Distributed Training
        """
        The training scripts (e.g., `train_caption.py`) use `torch.distributed.run`. 
        How is distributed training initialized and managed in the `main` function of these 
        scripts, with the help of functions from `utils.py`?
        """
    ]
    
    questions = [
    # High-Level Understanding
    "Based on the `README.md` and the overall file structure, can you provide a high-level overview of the BLIP (Bootstrapping Language-Image Pre-training) model? What are its key capabilities and the main problems it aims to solve?",
    "The `README.md` mentions several fine-tuned checkpoints for tasks like Image-Text Retrieval, Image Captioning, VQA, and NLVR2. Can you explain the differences in the model architecture or training process for each of these tasks, referencing the relevant training scripts (e.g., `train_retrieval.py`, `train_caption.py`)?",

    # Data and Configuration
    "In the `data/__init__.py` file, the `create_dataset` function uses different dataset classes for various tasks (e.g., `coco_karpathy_train`, `vqa_dataset`). Can you explain the key differences in how these datasets are structured and preprocessed, particularly in terms of how they handle image-text pairs and annotations?",
    "The `configs` directory contains several `.yaml` files (e.g., `caption_coco.yaml`, `retrieval_flickr.yaml`). Can you explain the role of these configuration files in the training and evaluation pipelines? What are some of the key hyperparameters defined in these files, and how do they affect the model's behavior?",

    # Model Architecture
    "The `models/blip.py` file defines the `BLIP_Base` and `BLIP_Decoder` classes. Can you provide a detailed explanation of the architecture of these two models? How do they leverage the Vision Transformer (ViT) and BERT models, and what is the role of the `med_config.json` file in defining the text encoder/decoder?",
    "In the `models/blip_itm.py` file, the `BLIP_ITM` model has two different match heads: 'itm' and 'itc'. Can you explain the difference between these two heads and how they are used for image-text matching? What do the 'itm_score' and 'itc_score' represent?",
    
    # Training and Fine-tuning
    "The `train_pretrain.py` script is used for pre-training the BLIP model. Can you explain the three loss functions used in this script: `loss_ita`, `loss_itm`, and `loss_lm`? What does each of these losses represent, and how do they contribute to the model's ability to understand the relationship between images and text?",
    "In the `train_caption.py` script, the `train` function uses a `cosine_lr_schedule`. Can you explain what a cosine learning rate schedule is and why it is a common choice for training deep learning models? How is it implemented in the `utils.py` file?",

    # Inference and Evaluation
    "The `predict.py` script provides a high-level interface for using the different BLIP models. Can you walk through the steps involved in making a prediction for the 'visual_Youtubeing' task, from loading the image to generating the final answer?",
    "The `eval_retrieval_video.py` script is used for zero-shot video-text retrieval. Can you explain how the model, which is pre-trained on images, is adapted for this task? How are video features extracted and compared with text features to compute the retrieval scores?",
    "In the `demo.ipynb` notebook, there are examples of both beam search and nucleus sampling for image captioning. Can you explain the difference between these two decoding methods and the effect of parameters like `num_beams`, `top_p`, `max_length`, and `min_length` on the generated captions?",

    # Code Implementation Details
    "The `models/vit.py` file contains the implementation of the Vision Transformer. Can you explain the purpose of the `interpolate_pos_embed` function and when it is used? Why is it necessary to interpolate positional embeddings when fine-tuning the model on a different image size?",
    "The `utils.py` file includes several utility functions for distributed training, such as `init_distributed_mode`, `get_world_size`, and `get_rank`. Can you explain the concept of distributed training and how these functions are used to enable it in the provided training scripts?",
    "In the `models/med.py` file, the `BertLMHeadModel` class is a modified version of the BERT model from the Hugging Face Transformers library. Can you identify some of the key modifications made to this class and explain their purpose in the context of the BLIP model?",
    "The `data/utils.py` file contains the `pre_caption` and `pre_question` functions. What kind of text preprocessing is performed by these functions, and why is it important for the model's performance?",
    "The `requirements.txt` file lists the Python packages required to run the code. Can you explain the role of each of these packages (e.g., `timm`, `transformers`, `fairscale`) in the project?",
    "The `models/blip_nlvr.py` script is designed for the NLVR2 task. Can you explain the unique architecture of the `BLIP_NLVR` model, particularly how it handles the two input images and the text query to produce a binary classification output?",
    "In `train_vqa.py`, the `vqa_collate_fn` is used in the DataLoader. What is the purpose of a collate function, and what specific processing does this particular function perform on the batch of data before it is fed to the model?",
    "The `README.md` provides links to download pre-trained and fine-tuned checkpoints. How are these checkpoints loaded into the models, and what is the significance of the `load_checkpoint` function in `models/blip.py`?"
]
    return questions 

def test_agent_and_save_results(agent, questions, output_file="agent_test_results.md"):
    """
    Tests the agent on a list of questions and saves the results to a markdown file.

    Args:
        agent: The initialized LangChain agent.
        questions: A list of strings, where each string is a question.
        output_file: The path to the markdown file to save results.
    """
    results_markdown = "# RAG Agent Test Results\n\n"
    results_markdown += "This document contains the test results for the RAG agent on 10 tricky questions about the `salesforce/BLIP` repository.\n\n"

    for i, question in enumerate(questions):
        log(f"Testing question {i+1}: {question}", prefix="Testing Agent")
        log(f"Query: {question}" , prefix="Testing Agent")

        # Get the answer from the agent
        answer = agent.invoke({
            "input": question
        })

        # Append to the markdown string
        results_markdown += f"## Question {i+1}: {question}\n\n" 
        results_markdown += "### Agent's Answer\n\n"
        results_markdown += f"{answer['output']}\n\n"

        if "intermediate_steps" in answer:
            results_markdown += "### Reasoning Trace\n\n"
            for step in answer["intermediate_steps"]:
                results_markdown += f"**Thought:** {step[0]}\n\n"
                results_markdown += f"**Observation:** {step[1]}\n\n"

        results_markdown += "---\n\n"

    # Save the final markdown content to a file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(results_markdown)
        log(f"\nSuccessfully saved test results to '{output_file}'" , prefix="File Saving")
    except IOError as e:
        print(f"\nError saving file: {e}")
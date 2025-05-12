# LLM-Edge-Distillation-Qwen-LLaMA3


## File structure(after downloading data below)
<pre> ML/
├── data_utils.ipynb            # Dataset loaders for sentiment, VQA, CIFAR-10, COCO/Flickr captions
├── download_model.py           # Downloads models (Qwen-7B, LLaMA-3-8B)
│
├── datasets/
│   ├── captioning/
│   │   ├── captioning_dataset/
│   │   │   └── Flicker8k_Dataset/       # Raw Flickr8k images
│   │   └── captioning_text/
│   │       ├── Flickr8k.token.txt       # Flickr8k image captions
│   │       ├── Flickr_8k.trainImages.txt
│   │       ├── Flickr_8k.testImages.txt
│   │       └── Flickr_8k.devImages.txt
│   ├── sentiment_test.csv               # Labeled text data for sentiment classification
│   ├── caption_test.csv                 # Flickr captions with image paths
│   └── ...
│
├── vqa_images/
│   └── validation/                      # VQA validation images (GQA)
│
├── cifar10_images/
│   └── train/                           # CIFAR-10 images as .png files
│
├── README.md
└── requirements.txt                    # packages needed to run the project

 </pre>


## Datasets

This project evaluates knowledge distillation and edge deployment using a set of benchmark datasets across multiple AI domains. Each dataset was selected to test the capabilities of large and distilled language models on tasks such as reasoning, perception, summarization, and generation.

### 1. Visual Question Answering (VQA)
- **Source**: [Stanford – Graphcore/gqa](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- **Task**: Answer natural language questions based on visual input.
- **Data Format**: Image-question pairs with multiple-choice answers.
- **Evaluation Metric**: Accuracy.

### 2. Image Classification (CIFAR-10)
- **Source**: [Kaggle CIFAR-10](https://www.kaggle.com/c/cifar-10/data)
- **Task**: Classify images into one of ten object categories.
- **Data Format**: 60,000 color images (32×32 pixels).
- **Evaluation Metric**: Classification accuracy.

### 3. Image Captioning
- **Source**: [Flicker8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Task**: Generate descriptive natural language captions for images.
- **Data Format**: Images paired with human-written captions.
- **Evaluation Metric**: BLEU, ROUGE (optional, task-dependent).

### 4. Text Summarization
- **Source**: [Kaggle Notebook – Text Summarization](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models)
- **Task**: Generate concise summaries of longer text passages.
- **Data Format**: Document-summary pairs.
- **Evaluation Metrics**:
  - **ROUGE-L**: Measures longest common subsequence.
  - **ROUGE-S**: Measures skip-bigram overlap.

### 5. Sentiment Analysis
- **Source**: [Kaggle – Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- **Task**: Classify text into positive, negative, or neutral sentiments.
- **Data Format**: Text with corresponding sentiment label.
- **Evaluation Metrics**: Accuracy, F1-score.

### 6. Code Generation (HumanEval)
- **Source**: [DataCamp HumanEval Tutorial](https://www.datacamp.com/tutorial/humaneval-benchmark-for-evaluating-llm-code-generation-capabilities)
- **Task**: Generate functional code to solve algorithmic problems.
- **Data Format**: Prompt with function signature, docstring, and unit tests.
- **Evaluation Metric**: Pass@k — proportion of generated programs that pass all unit tests.



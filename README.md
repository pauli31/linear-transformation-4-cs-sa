# Linear Transformations for Cross-lingual Sentiment Analysis
This is the repository for the papers: 

#### Linear Transformations for Cross-lingual Sentiment Analysis
Accepted to [TSD](https://www.tsdconference.org/tsd2022/) conference

#### Are the Multilingual Models Better? Improving Czech Sentiment with Transformers
Accepted to [RANLP 2021](https://aclanthology.org/2021.ranlp-1.128/) conference

for the multilingual Transformers-based approach, use the corresponding [GitHub repository](https://github.com/pauli31/improving-czech-sentiment-transformers).
for prompting source codes, use the corresponding [GitHub repository](https://github.com/pauli31/czech-sentiment-prompting)

#### A Comprehensive Study of Cross-lingual Sentiment Analysis
Submitted (under review) to [Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications)


We use linear transformations for the cross-lingual sentiment analysis between Czech, English and French. 

## Data 

#### Word Embeddings 
Download Czech (cc.cs.300.bin), French (cc.fr.300.bin) and English (cc.end.300.bin) fasttext embeddings from https://fasttext.cc/docs/en/crawl-vectors.html

copy into 
```
 ./data/embeddings
```

The word embeddings (fasttext_csfd.bin, fasttext_en.bin, fasttext_fr.bin) pre-trained by us can be downloaded from our [disk](https://drive.google.com/drive/folders/1HBIt9UaJzm3gl8px8HmWdloE0AgIFWQP?usp=sharing)
copy them into 
```
 ./data/embeddings
```

#### Dictionaries
required for the linear transformations can be downloaded from our [disk](https://drive.google.com/drive/folders/1ko0OVUjgi0h1oriracGvPkMqw8g9zHB8?usp=sharing) 

copy them into 
```
 ./data/dictionary
```

#### Polarity Data
Download the original [Czech datasets](http://nlp.kiv.zcu.cz/research/sentiment) ([CSFD](http://nlp.kiv.zcu.cz/data/research/sentiment/csfd.zip)) 
Download the original [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  

The splits for these datasets can be obtained from the corresponding [GitHub repository](https://github.com/pauli31/improving-czech-sentiment-transformers)

Put the resulted splits into polarity folder
copy them into 
```
 ./data/polarity
```

[SST Dataset](https://nlp.stanford.edu/sentiment/)
[Allocine Dataset](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

Please contact us for the corresponding splits of these datasets.



Setup:
--------


1) #### Clone github repository 
   ```
   git clone git@github.com:pauli31/linear-transformation-4-cs-sa.git
   ```
2) #### Setup conda environment

   Check version
    ```
    # print version
    conda -V
   
    # print available enviroments
    conda info --envs
    ```
    Create conda enviroment
   
    ```
    # create enviroment 
    conda create --name cross-lingual-transformation-sentiment python=3.7 -y
    
    # check that it was created
    conda info --envs
   
    # activate enviroment
    conda activate cross-lingual-transformation-method
   
    # see already installed packages
    pip freeze  
    ```
   
   Install requirements
   ```
   pip install -r requirements.txt
   ```

    
3) #### Install dependencies
   Install requirements
   ```
   pip install -r requirements.txt
   ```
   
   Install our library for linear-transformations from the [GitHub repository](https://github.com/pauli31/cross-lingual-transform-method)
   ```
   cd  kiv-nlp-cross-lingual-transformations
   pip install .
   # or
   pip install --upgrade .
   # or
   pip install -v --upgrade .
   ```


Usage
--------
#### Monolingual Baselines
you can use the 
```
--use_cpu
```
parameter to run on CPU

LSTM Baseline
Default Embeddings
```
python3 main.py --model_name lstm --dataset_name imdb  --embeddings fasttext_en.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name lstm --dataset_name sst  --embeddings fasttext_en.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name lstm --dataset_name csfd  --embeddings fasttext_csfd.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name lstm --dataset_name allocine  --embeddings fasttext_fr.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
```

Original fasttext embeddings
```
python3 main.py --model_name lstm --dataset_name imdb  --embeddings cc.en.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name lstm --dataset_name sst  --embeddings cc.en.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name lstm --dataset_name csfd  --embeddings cc.cs.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name lstm --dataset_name allocine  --embeddings cc.fr.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --num_layers 2 --bidirectional --max_seq_len 0 --use_random_seed --scheduler constant
```

CNN Baseline
```
python3 main.py --model_name cnn --dataset_name imdb  --embeddings fasttext_en.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 8 --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name cnn --dataset_name sst  --embeddings fasttext_en.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name cnn --dataset_name csfd  --embeddings fasttext_csfd.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name cnn --dataset_name allocine  --embeddings fasttext_fr.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
```

Original fasttext embeddings
```
python3 main.py --model_name cnn --dataset_name imdb  --embeddings cc.en.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name cnn --dataset_name sst  --embeddings cc.en.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name cnn --dataset_name csfd  --embeddings cc.cs.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
python3 main.py --model_name cnn --dataset_name allocine  --embeddings cc.fr.300.bin --embeddings_type fasttext --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant
```

you can add the normalization params
```
--normalize_before
--normalize_after
```

#### Cross-lingual experiments
To perform cross-lingual experiments you firstly have to select the transformation using the **transformation** parameter
```
--transformation [lst | orthogonal | cca | ranking | orto-ranking]
```

and set the cross-lingual mode
```
--mode crosslingual
```

Next the normalization parameter can be added

To perform normalization before 
```
--normalize_before
```

To perform normalization after
```
--normalize_after
```

Next the dataset pair must be selected where the first name of the dataset denotes the **source** language and the second dataset name
denotes the **target** language, for example pair csfd-imdb. The csfd dataset will be used for training and model will be evaluated on the imdb dataset
The corresponding embeddings must be set. by the **--embeddings** and **--target_embeddings** parameters.
For example:
```
--embeddings cc.cs.300.bin --target_embeddings cc.en.300.bin
```

The direction of the transformation is given by the **--transformation_type** parameter

```
--transformation_type [target_to_source | source_to_target]
```

So for example for training a model on **csfd** data and evaluate it on the english **imdb dataset**. 
With normalization before and after with original fasttext embeddings, and with the linear transformation direction from source (czech) to
target word embeddings (english), with **orthogonal** transformation the run command could look as follows :
```
python3 main.py --model_name cnn --dataset_name csfd-imdb  --embeddings cc.cs.300.bin --embeddings_type fasttext --target_embeddings cc.en.300.bin --mode crosslingual --binary --lowercase --batch_size 32 --epoch_num 10 --max_seq_len 0 --use_random_seed --scheduler constant --transformation orthogonal --normalize_before --normalize_after
```



Analogies
--------
The source code for cross-lingual word analogies can be found in the corresponding [GitHub repository](https://github.com/pauli31/cross-lingual-transform-method).

License:
--------
The dataset and code can be freely used for academic and research purposes.
It is strictly prohibited to use the dataset for any commercial purpose.

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Publication:
--------

If you use our dataset, software or approach for academic research, please cite the our [TSD](https://www.tsdconference.org/tsd2022/), [RANLP 2021](https://aclanthology.org/2021.ranlp-1.128/) .

```
@inproceedings{priban-tsd-2022,
    author="P{\v{r}}ib{\'a}{\v{n}}, Pavel
    and {\v{S}}m{\'i}d, Jakub
    and Mi{\v{s}}tera, Adam
    and Kr{\'a}l, Pavel",
    editor="Sojka, Petr
    and Hor{\'a}k, Ale{\v{s}}
    and Kope{\v{c}}ek, Ivan
    and Pala, Karel",
    title="Linear Transformations for Cross-lingual Sentiment Analysis",
    booktitle="Text, Speech, and Dialogue",
    year="2022",
    publisher="Springer International Publishing",
    address="Cham",
    pages="125--137",
    isbn="978-3-031-16270-1"
}

@inproceedings{priban-steinberger-2021-multilingual,
    title = "Are the Multilingual Models Better? Improving {C}zech Sentiment with Transformers",
    author = "P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
      Steinberger, Josef",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)",
    month = sep,
    year = "2021",
    address = "Held Online",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/2021.ranlp-main.128",
    pages = "1138--1149"
    }

```

Contact:
--------
pribanp@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)

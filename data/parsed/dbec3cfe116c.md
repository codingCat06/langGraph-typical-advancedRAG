# TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision

Yunyi Zhang University of Illinois Urbana-Champaign Urbana, IL, USA yzhan238@illinois.edu Ruozhen Yang\* University of Illinois Urbana-Champaign Urbana, IL, USA ruozhen2@illinois.edu Xueqiang Xu\* University of Illinois Urbana-Champaign Urbana, IL, USA xx19@illinois.edu Rui Li\*
University of Science and
Technology of China
Hefei, China
rui\_li@mail.ustc.edu.cn

Jinfeng Xiao University of Illinois Urbana-Champaign Urbana, IL, USA jxiao13@illinois.edu

Jiaming Shen Google Deepmind New York, NY, USA jmshen@google.com Jiawei Han University of Illinois Urbana-Champaign Urbana, IL, USA hanj@illinois.edu

#### Abstract

Hierarchical text classification aims to categorize each document into a set of classes in a label taxonomy, which is a fundamental web text mining task with broad applications such as web content analysis and semantic indexing. Most earlier works focus on fully or semi-supervised methods that require a large amount of human annotated data which is costly and time-consuming to acquire. To alleviate human efforts, in this paper, we work on hierarchical text classification with a minimal amount of supervision: using the sole class name of each node as the only supervision. Recently, large language models (LLM) have shown competitive performance on various tasks through zero-shot prompting, but this method performs poorly in the hierarchical setting because it is ineffective to include the large and structured label space in a prompt. On the other hand, previous weakly-supervised hierarchical text classification methods only utilize the raw taxonomy skeleton and ignore the rich information hidden in the text corpus that can serve as additional class-indicative features. To tackle the above challenges, we propose TELEClass, Taxonomy Enrichment and LLM-Enhanced weakly-supervised hierarchical text Classification, which combines the general knowledge of LLMs and task-specific features mined from an unlabeled corpus. TELEClass automatically enriches the raw taxonomy with class-indicative features for better label space understanding and utilizes novel LLM-based data annotation and generation methods specifically tailored for the hierarchical setting. Experiments show that TELEClass can significantly outperform previous baselines while achieving comparable performance to zero-shot prompting of LLMs with drastically less inference cost.

### **CCS Concepts**

• Information systems  $\to$  Data mining; • Computing methodologies  $\to$  Natural language processing.

\*Equal Contribution

![](_page_0_Picture_14.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License WWW '25, Sydney, NSW, Australia

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1274-6/25/04

https://doi.org/10.1145/3696410.3714940

<span id="page-0-0"></span>![](_page_0_Picture_18.jpeg)

Figure 1: An example document tagged with 3 classes. We automatically enrich each node with class-indicative terms and utilize LLMs to facilitate classification.

#### **Keywords**

Weakly-Supervised Text Classification, Hierarchical Text Classification, Taxonomy Enrichment, Large Language Model

#### **ACM Reference Format:**

Yunyi Zhang, Ruozhen Yang\*, Xueqiang Xu\*, Rui Li\*, Jinfeng Xiao, Jiaming Shen, and Jiawei Han. 2025. TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision. In *Proceedings of the ACM Web Conference 2025 (WWW '25), April 28-May 2, 2025, Sydney, NSW, Australia.* ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3696410.3714940

#### 1 Introduction

Hierarchical text classification, aiming to classify documents into one or multiple classes in a label taxonomy, is a fundamental task in web text mining and NLP. Compared with standard text classification where label space is flat and relatively small (e.g., less than 20 classes), hierarchical text classification is more challenging given the larger and more structured label space and the existence of fine-grained and long-tail classes. Hierarchical text classification has broad applications such as web content organization [13], semantic indexing [25, 29], and query classification [7, 21, 23]. Recent studies also show that hierarchically structured text [14, 43] and document-level tagging [39] can improve retrieval-augmented generation for large language models.

The key challenge of hierarchical text classification is how to understand the large structured label space to distinguish the semantics of similar classes. Most earlier works tackle this task in fully supervised [9, 24, 58] or semi-supervised settings [5, 20], and different models are proposed to learn from a substantial amount of human-labeled data. However, acquiring human annotation is often costly, time-consuming, and not scalable.

Recently, large language models (LLM) such as GPT-4 [37] and Claude 3 [2] have demonstrated strong performance in flat text classification [49]. However, applying LLMs in hierarchical settings remains challenging [59]. Directly including hundreds of classes in prompts is ineffective and inefficient, leading to structural information loss, diminished clarity for LLMs at distinguishing class-specific information, and prohibitively expensive inference cost given the long prompt for each test document.

Along another line of research, Meng et al. [35] propose to train a moderate-size text classifier by utilizing a small set of keywords or labeled documents for each class and a large unlabeled corpus. However, compiling keyword lists for hundreds of classes and obtaining representative documents for each specific and niche category still demand significant human efforts. Shen et al. [45] study the hierarchical text classification with *minimal supervision*, which takes the class name as the only supervision signal. Specifically, they introduce TaxoClass which generates pseudo labels with a textual entailment model for classifier training. However, this method overlooks additional class-relevant features in the corpus that could be helpful for label space understanding. It also suffers from the unreliable pseudo label selection because the entailment model is not trained to compare which class is more relevant to the document.

In this study, we advance minimally supervised hierarchical text classification by taking the advantage of both LLMs' text understanding ability and task-specific knowledge of the unlabeled text corpus. First, we tackle the challenge of label space understanding by enriching the label taxonomy with class-specific terms derived from two sources: LLM generation and automated extraction from the corpus. For example, the "conditioner" class in Figure 1 is enriched with key terms like "moisture" and "soft hair", which distinguish it from other classes. These terms enhance the supervision signal by combining knowledge from LLMs and text corpus and improve the pseudo label quality for classifier training. Second, we improve LLMs' ability in hierarchical text classification from two perspectives: we enhance LLM annotation efficiency and effectiveness through a taxonomy-guided candidate search and also optimize LLM-based document generation to create more precise pseudo data by using taxonomy paths.

Leveraging the above ideas, we introduce TELEClass: <u>Taxonomy Enrichment</u> and <u>LLM-Enhanced</u> weakly-supervised hierarchical text <u>Classification</u>. TELEClass consists of four major steps: (1) *LLM-Enhanced Core Class Annotation*, where we identify document "core classes" (i.e., fine-grained classes that most accurately describe the documents) by first enriching the taxonomy with LLM-generated key terms and then finding candidate classes with a top-down tree search algorithm for LLM to select the most precise core classes. (2) *Corpus-Based Taxonomy Enrichment*, where we analyze the taxonomy structure to additionally identify class-indicative topical terms through semantic and statistical analysis on the corpus. (3) *Core* 

Class Refinement with Enriched Taxonomy, where we embed documents and classes based on the enriched label taxonomy and refined the initially selected core classes by identifying the most similar classes for each document. (4) Text Classifier Training with Path-Based Data Augmentation, where we sample label paths from the taxonomy and guide the LLM to generate pseudo documents most accurately describing these fine-grained classes. Finally, we train the text classifier on two types of pseudo labels, the core classes and the generated data, with a simple text matching network and multi-label training strategy.

The contributions of this paper are summarized as follows:

- We propose TELEClass, a new method for minimally supervised hierarchical text classification, which requires only the class names of the label taxonomy as supervision to train a multi-label text classifier.
- We propose to enrich the label taxonomy with class-indicative terms, based on which we utilize an embedding-based documentclass matching method to improve the pseudo label quality.
- We study two ways of adopting large language models to hierarchical text classification, which can improve the pseudo label quality and solve the data scarcity issue for fine-grained classes.
- Experiments on two datasets show that TELEClass can significantly outperform zero-shot and weakly-supervised hierarchical text classification baselines, while also achieving comparable performance to GPT-4 with drastically less inference cost. <sup>1</sup>

#### 2 Problem Definition

The minimally-supervised hierarchical text classification task aims to train a text classifier that can categorize each document into multiple nodes on a label taxonomy by using the name of each node as the only supervision [45]. For example, in Figure 1, the input document is classified as "hair care", "shampoo", and "scalp treatment".

Formally, the task input includes an unlabeled text corpus  $\mathcal{D}=\{d_1,\ldots,d_{|\mathcal{D}|}\}$  and a directed acyclic graph (DAG)  $\mathcal{T}=(C,\mathcal{R})$  as the label taxonomy. Each  $c_i\in C$  represents a target class in the taxonomy, coupled with a unique textual surface name  $s_i$ . Each edge  $\langle c_i,c_j\rangle\in\mathcal{R}$  indicates a hypernymy relation, where class  $c_j$  is a subclass of  $c_i$ . For example, one such edge in Figure 1 is between  $s_i=$  "hair care" and  $s_j=$  "shampoo". Then, the goal of our task is to train a multi-label text classifier  $f(\cdot)$  that can map a document d into a binary encoding of its corresponding classes,  $f(d)=[y_1,\ldots,y_{|C|}],$  where  $y_i=1$  represents that d belongs to class  $c_i$ , otherwise  $y_i=0$ .

We assume the label taxonomy to be a DAG instead of a tree, because it aligns better with real applications as one node can have multiple parents with different meanings. Therefore, the classifier needs to assign a document multiple labels in different levels and paths. Also, this setting does not restrict the prediction to be one label per level. In this way, the output can consist of labels from multiple paths or not necessarily contain a leaf node if the document is not specifically discussing a fine-grained topic. Such a setting is more flexible to fit various applications and also a more challenging research task.

<span id="page-1-0"></span><sup>&</sup>lt;sup>1</sup>Our code and datasets can be found at: https://github.com/yzhan238/TFLFClass

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

Figure 2: Overview of the TELEClass framework.

# 3 Methodology

In this section, we will introduce TELEClass consisting of the following modules: (1) LLM-enhanced core class annotation, (2) corpusbased taxonomy enrichment, (3) core class refinement with enriched taxonomy, and (4) text classifier training with path-based data augmentation. Figure 2 shows an overview of TELEClass.

# <span id="page-2-2"></span>3.1 LLM-Enhanced Core Class Annotation

Inspired by previous studies, we first tag each document with its "core classes", which are defined as a set of classes that can describe the document most accurately [45]. This process also mimics the process of human performing hierarchical text classification: first select a set of most essential classes for the document and then trace back to their relevant classes to complete the labeling. For example, in Figure 1, by first tagging the document with "shampoo" and "scalp treatment", we can easily find its complete set of classes.

In this work, we propose to enhance the core class annotation process of previous methods with the power of LLMs. To utilize LLMs for core class annotation, we apply a structure-aware candidate core class selection method to reduce the label space for each document. This step is necessary because LLMs can hardly comprehend a large, structured hierarchical label space that cannot be easily represented in a prompt. We first define a similarity score between a class and a document that will be used in candidate selection. To better capture the semantics, we propose to use LLMs to generate a set of class relevant keywords to enrich the raw taxonomy structure and consolidate the meaning of each class. For example, "shampoo" and "conditioner" are two fine-grained classes that are similar to each other. We can effectively separate the two classes by identifying a set of class-specific terms such as "flakes" for "shampoo" and "moisture" for "conditioner". We prompt an LLM to enrich the raw label taxonomy with a set of key terms for each class, denoted as  $T_c^{\text{LLM}}$  for the class c. To ensure the generated terms can uniquely identify c, we ask the LLM to generate terms that are relevant to c and its parent while irrelevant to siblings of c. With this enriched set of terms for each class, we define the similarity score between a document d and a class c as the maximum cosine similarity with the key terms:

$$sim(c,d) = \max_{t \in T_c^{\rm LLM}} \cos(\vec{t}, \vec{d}), \tag{1}$$

where  $\vec{\circ}$  denotes a vector representation by a pre-trained semantic encoder (e.g., Sentence Transformer [41]).

This newly defined similarity measure is then used for candidate core class selection. Given a document, we start from the root node at level l=0, select the l+3 most similar children classes to the document at level l using the similarity score defined above, and continue to the next level with only the selected classes. The increasing number of selected nodes accounts for the growing number of classes when going deeper into the taxonomy. Finally, all the classes ever selected in this process will be the candidate core classes for this document, which share the most similarity with the document according to the label hierarchy.  $^2$ 

Finally, we instruct an LLM to select the core classes for each document from the selected candidates, which produces an initial set of core classes (denoted as  $\mathbb{C}^0_i$ ) for each document  $d_i \in \mathcal{D}$ .

### <span id="page-2-3"></span>3.2 Corpus-Based Taxonomy Enrichment

In the previous step, we enrich the raw taxonomy structure with LLM-generated key terms, which are derived from the general knowledge of LLMs but may not accurately reflect the corpus-specific knowledge. Therefore, we propose further enriching the classes with class-indicative terms mined from the text corpus. By doing this, we can combine the general knowledge of LLMs and corpus-specific knowledge to better enhance the very weak supervision, which is essential for correctly understanding fine-grained classes that are hard to distinguish. Formally, given a class  $c \in C$  and its siblings corresponding to one of its parents  $c_p$ ,  $Sib(c, c_p) = \{c' \in C | \langle c_p, c' \rangle \in \mathcal{R} \}$ ,  $c_p \in Par(c)$ , we find a set of corpus-based class-indicative terms of c corresponding to  $c_p$ , denoted as  $T(c, c_p) = \{t_1, t_2, \ldots, t_k\}$ . Each term in  $T(c, c_p)$  can signify the class c and distinguish it from its siblings under  $c_p$ .

We first collect a set of relevant documents  $D_c^0 \subset \mathcal{D}$  for each class c, which contains all the documents whose initial core classes contain c or its descendants. Then, inspired by [50, 67], we consider the following three factors for class-indicative term selection and adapt them to the hierarchical setting.

Popularity: a class-indicative term t of a class c should be frequently mentioned by its relevant documents, which is quantified

<span id="page-2-1"></span> $<sup>^2</sup>$ Refer to Shen et al. [45] for more details on the tree search algorithm.

by the log normalization of its document frequency,

$$pop(t,c) = \log(1 + df(t, D_c^0)),$$
 (2)

where df(t, D) stands for the number of documents in D that mention t.

 Distinctiveness: a class-indicative term t for a class c should be infrequent in its siblings, which is quantified as the softmax of BM25 relevance function [42] over the set of siblings,

$$dist(t,c,c_p) = \frac{\exp(BM25(t,D_c^0))}{1 + \sum_{c' \in Sib(c,c_p)} \exp(BM25(t,D_{c'}^0))}. \tag{3}$$

• *Semantic similarity*: a class-indicative term *t* should also be semantically similar to the class name of *c*, which is quantified as the cosine similarity between their embeddings derived from a pre-trained encoder (e.g., BERT [10]), denoted as sem(c, t).

Finally, we define the *affinity* score between a term t and a class c corresponding to parent p to be the geometric mean of the above scores, denoted as  $aff(t, c, c_p)$ .

To enrich the taxonomy, we first apply a phrase mining tool, AutoPhrase [44], to mine quality single-token and multi-token phrases from the corpus as candidate terms<sup>3</sup>. Then, for each class c and each of its parents  $c_p$ , we select the top-k terms with the highest affinity scores with c corresponding to p, denoted as  $T(c, c_p)$ . Then, we take the union of these corpus-based terms together with the LLM-generated terms in the previous step to get the final enriched class-indicative terms for class c,

<span id="page-3-2"></span>
$$T_{c} = \left(\bigcup_{c_{p} \in Par(c)} T(c, c_{p})\right) \bigcup T_{c}^{\text{LLM}}.$$
 (4)

# <span id="page-3-1"></span>3.3 Core Class Refinement with Enriched Taxonomy

With the enriched class-indicative terms for each class, we propose to further utilize them to refine the initial core classes. In this paper, we adopt an embedding-based document-class matching method. Unlike previous methods in flat text classification [53] that use keyword-level embeddings to estimate document and class representations, here, we are able to define class representations directly based on document-level embeddings thanks to the rough class assignments we created in the core class annotation step (c.f. Sec. 3.1).

To obtain document representations, we utilize a pre-trained Sentence Transformer model [41] to encode the entire document, which we denote as  $\vec{d}$ . Then, for each class c, we identify a subset of its assigned documents that explicitly mention at least one of the class-indicative keywords and thus most confidently belong to this class,  $D_c = \{d \in D_c^0 | \exists w \in T_c, w \in d\}$ . Then, we use the average of their document embeddings as the class representation,  $\vec{c} = \frac{1}{|D_c|} \sum_{d \in D_c} \vec{d}$ . Finally, we compute the document-class matching score as the cosine similarity between their representations.

Based on the document-class matching scores, we make an observation that the true core classes often have much higher matching scores with the document compared to other classes. Therefore, we use the largest "similarity gap" for each document to identify its core classes. Specifically, for each document  $d_i \in \mathcal{D}$ , we first get

a ranked list of classes according to the matching scores, denoted as  $[c_1^i,c_2^i,\ldots,c_{|C|}^i]$ , where  $\mathrm{diff}^i(j) \coloneqq \cos(\vec{d}_i,\vec{c}_j^i) - \cos(\vec{d}_i,\vec{c}_{j+1}^i) > 0$  for  $j \in \{1,\ldots,|C|-1\}$ . Then, we find the position  $m_i$  with the highest similarity difference with its next one in the list. After that, we treat the classes ranked above this position as this document's refined core classes  $\mathbb{C}_i$ , and the corresponding similarity gap as the confidence estimation  $conf_i$ .

<span id="page-3-3"></span>
$$conf_{i} = \operatorname{diff}^{i}(m_{i}), \quad \mathbb{C}_{i} = \{c_{1}^{i}, \dots, c_{m_{i}}^{i}\},$$

$$m_{i} = \underset{j \in \{1, \dots, |C|-1\}}{\operatorname{arg max}} \operatorname{diff}^{i}(j).$$
(5)

Finally, we select top 75% of documents  $d_i$  and their refined core classes with the highest confidence scores  $con f_i$ , denoted as  $\mathcal{D}^{core}$ .

# 3.4 Text Classifier Training with Path-Based Data Augmentation

The final step of TELEClass is to train a hierarchical text classifier using the confident refined core classes. One straightforward way is to directly use the selected core classes as a complete set of pseudo-labeled documents and train a text classifier in a common supervised way. However, such a strategy is ineffective, because the core classes are not comprehensive enough and cannot cover all the classes in the taxonomy. This is because the hierarchical label space naturally contains fine-grained and long-tail classes, and they are often not guaranteed to be selected as core classes due to their low frequency. Empirically, for the two datasets we use in our experiments, Amazon and DBPedia, the percentages of classes never selected as core classes are 11.6% and 5.4%, respectively. These missing classes will never be used as positive classes in the training process if we only train the classifier with the selected core classes.

Therefore, to overcome this issue, we propose the idea of pathbased document generation by LLMs to generate a small number of augmented documents (e.g., q = 5) for each distinct path from a level-1 node to a leaf node in the taxonomy. By adding the generated documents to the pseudo-labeled data, we can ensure that each class of the taxonomy will be a positive class of at least q documents. Because we generate a small constant number of documents for each label path, it also does not affect the distribution of the frequent classes. Moreover, we use a path instead of a single class to guide the LLM generation, because the meaning of lower-level classes is often conditioned on their parents. For example, in Figure 2, a path "hair care" → "shampoo" can guide the LLM to generate text about hair shampoo instead of pet shampoo or carpet shampoo that are in different paths. To promote data diversity, we make one LLM query for each path and ask it to generate q diverse documents. We denote the generated documents as  $\mathcal{D}^{\mathrm{gen}}$ . Appx. B shows the prompts we used.

Now, with two sets of data, the pseudo-labeled documents  $\mathcal{D}^{\text{core}}$  and LLM-generated documents  $\mathcal{D}^{\text{gen}}$ , we are ready to introduce the classifier architecture and the training process.

Classifier architecture. We use a simple text matching network similar to [45] as our model architecture, which includes a document encoder initialized with a pre-trained BERT-base model [10] and a log-bilinear matching network. Class representations are initialized by class name embeddings (c.f. Sec. 3.2) and are detached from the encoder model, so only the embeddings will be updated without

<span id="page-3-0"></span> $<sup>^3</sup>$ Our method is flexible with any kinds of phrase mining methods like Gu et al. [19].

Table 1: Datasets overview.

<span id="page-4-0"></span>

| 531<br>298 |
|------------|
|            |

back-propagation to the backbone model. Formally, the classifier predicts the probability of document  $d_i$  belonging to class  $c_i$ :

$$p(c_j|d_i) = \mathcal{P}(y_j = 1|d_i) = \sigma(\exp(\mathbf{c}_i^T \mathbf{W} \mathbf{d}_i)),$$

where  $\sigma$  is the sigmoid function, **W** is a learnable interaction matrix, and  $\mathbf{c}_i$  and  $\mathbf{d}_i$  are the encoded class and document.

**Training process.** For each document  $d_i$  tagged with core classes  $\mathbb{C}_i$ , we construct its positive classes  $\mathbb{C}_{i,+}^{\text{core}}$  as the union of its core classes and their ancestors in the label taxonomy, and its negative classes  $\mathbb{C}_{i,-}^{\text{core}}$  are the ones that are not positive classes or descendants of any core class. This is because the ancestors of confident core classes are also likely to be true labels, and the descendants may not all be negative given that the automatically generated core classes are not optimal. Formally,

$$\begin{split} \mathbb{C}_{i,+}^{\text{core}} &= \mathbb{C}_i \cup \left( \cup_{c \in \mathbb{C}_i} Anc(c) \right), \\ \mathbb{C}_{i,-}^{\text{core}} &= C - \mathbb{C}_{i,+}^{\text{core}} - \cup_{c \in \mathbb{C}_i} Des(c), \end{split}$$

where Anc(c) and Des(c) denote the set of ancestors and descendants and class c, respectively. For the LLM-generated documents, we are more confident in its pseudo labels, so we simply treat all the classes in the corresponding path as positive classes and all other classes as negative,

$$\mathbb{C}_{p,+}^{\mathrm{gen}} = \mathbb{C}_p, \quad \mathbb{C}_{p,-}^{\mathrm{gen}} = C - \mathbb{C}_p.$$

Then, we train a multi-label classifier with the binary cross entropy loss (BCE):

$$\mathcal{L}^{\text{core}} = -\sum_{d_i \in \mathcal{D}^{\text{core}}} \left( \sum_{c_j \in \mathbb{C}_{i,+}^{\text{core}}} \log p(c_j|d_i) + \sum_{c_j \in \mathbb{C}_{i,-}^{\text{core}}} \log \left(1 - p(c_j|d_i)\right) \right)$$

$$\mathcal{L}^{\text{gen}} = -\sum_{d_i^p \in \mathcal{D}^{\text{gen}}} \left( \sum_{c_j \in \mathbb{C}_{p,+}^{\text{gen}}} \log p(c_j | d_i^p) + \sum_{c_j \in \mathbb{C}_{p,-}^{\text{gen}}} \log \left(1 - p(c_j | d_i^p)\right) \right)$$

$$\mathcal{L} = \mathcal{L}^{\text{core}} + \frac{|\mathcal{D}^{\text{core}}|}{|\mathcal{D}^{\text{gen}}|} \cdot \mathcal{L}^{\text{gen}}$$
 (6)

The loss terms of two sets of data are weighted by their relative size,  $\frac{|\mathcal{D}^{core}|}{|\mathcal{D}^{gen}|}$ . Notice that we do not continue training the classifier with self-training that is commonly used in previous studies [35, 45]. Using self-training may further improve the model performance, which we leave for future exploration. Algorithm 1 in Appendix summarizes TELEClass.

#### 4 Experiments

### 4.1 Experiment Setup

- *4.1.1 Datasets.* We use two public datasets in different domains for evaluation. Table 1 shows the data statistics.
- Amazon-531 [32] consists of Amazon product reviews and a three-layer label taxonomy of product types.

- DBPedia-298 [28] consists of Wikipedia articles with a threelayer label taxonomy of its categories.
- 4.1.2 Compared Methods. We compare the following methods on the weakly-supervised hierarchical text classification task.
- **Hier-0Shot-TC** [57] is a zero-shot approach, which utilizes a pretrained textual entailment model to iterative find the most similar class at each level for a document.
- **GPT-3.5-turbo** is a zero-shot approach that queries GPT-3.5-turbo by directly providing all classes in the prompt.
- Hier-doc2vec [27] is a weakly-supervised approach, which first trains document and class representations in a same embedding space, and then iteratively selects the most similar class at each level.
- WeSHClass [35] is a weakly-supervised approach using a set of keywords for each class. It first generates pseudo documents to pretrain text classifiers and then performs self-training.
- TaxoClass [45] is a weakly-supervised approach that only uses
  the class name of each class. It first uses a textual entailment
  model with a top-down search and corpus-level comparison to select core classes, which are then used as pseudo training data. We
  include both its full model, TaxoClass, and its variation TaxoClassNoST that does not apply self-training on the trained classifier,
  which is the same as TELEClass.
- TELEClass is our newly proposed weakly-supervised approach that only uses the class name for each class.
- Fully-Supervised is a fully-supervised baseline that uses the entire labeled training data to train the text matching network used in TELEClass.
- 4.1.3 Evaluation Metrics. Following previous studies [45], we utilize the following evaluation metrics: Example-F1, Precision at k (P@k) for k = 1, 3, and Mean Reciprocal Rank (MRR). See Appendix C for detailed definitions.
- <span id="page-4-1"></span>4.1.4 Implementation Details. We use Sentence Transformer [41] all-mpnet-base-v2 as the text encoder for the similarity measure in Section 3.1 and Section 3.3. We query GPT-3.5-turbo-0125 for LLM-based taxonomy enrichment, core class annotation, and path-based generation. For corpus-based taxonomy enrichment, we get the term and class name embeddings using a pre-trained BERT-base-uncased [10], we select top k=20 enriched terms for each class (c.f. Section 3.2). We generate q=5 documents with path-based generation for each class. The document encoder in the final classifier is initialized with BERT-base-uncased for a fair comparison with the baselines. We train the classifier using AdamW optimizer with a learning rate 5e-5, and the batch size is 64. The experiments are run on one NVIDIA RTX A6000 GPU.

#### 4.2 Experimental Results

Table 2 shows the evaluation results of all the compared methods. We make the following observations. (1) Overall, TELEClass achieves significantly better performance than other strong zeroshot and weakly-supervised baselines, which demonstrates the effectiveness of TELEClass on the hierarchical text classification task without any human supervision. (2) By comparing with other weakly-supervised methods, we find that TELEClass significantly

<span id="page-5-0"></span>Table 2: Experiment results on Amazon-531 and DBPedia-298 datasets, evaluated by Example-F1, P@k, and MRR. The best score among zero-shot and weakly-supervised methods is boldfaced. "†" indicates the numbers for these baselines are directly from previous paper [45]. "—" means the method cannot generate a ranking of predictions and thus MRR cannot be calculated.

| Supervision Type   | Methods                     |            | Amazon | -531   |        | DBPedia-298 |        |        |        |
|--------------------|-----------------------------|------------|--------|--------|--------|-------------|--------|--------|--------|
| ouper vision 1, po | 1/10011000                  | Example-F1 | P@1    | P@3    | MRR    | Example-F1  | P@1    | P@3    | MRR    |
| Zero-Shot          | Hier-0Shot-TC <sup>†</sup>  | 0.4742     | 0.7144 | 0.4610 | _      | 0.6765      | 0.7871 | 0.6765 | _      |
| Zero-Snot          | ChatGPT                     | 0.5164     | 0.6807 | 0.4752 | _      | 0.4816      | 0.5328 | 0.4547 | _      |
|                    | Hier-doc2vec <sup>†</sup>   | 0.3157     | 0.5805 | 0.3115 | _      | 0.1443      | 0.2635 | 0.1443 | _      |
|                    | WeSHClass <sup>†</sup>      | 0.2458     | 0.5773 | 0.2517 | _      | 0.3047      | 0.5359 | 0.3048 | _      |
| Weakly-Supervised  | TaxoClass-NoST <sup>†</sup> | 0.5431     | 0.7918 | 0.5414 | 0.5911 | 0.7712      | 0.8621 | 0.7712 | 0.8221 |
|                    | TaxoClass <sup>†</sup>      | 0.5934     | 0.8120 | 0.5894 | 0.6332 | 0.8156      | 0.8942 | 0.8156 | 0.8762 |
|                    | TELEClass                   | 0.6483     | 0.8505 | 0.6421 | 0.6865 | 0.8633      | 0.9351 | 0.8633 | 0.8864 |
| Fully-Supervised   |                             | 0.8843     | 0.9524 | 0.8758 | 0.9085 | 0.9786      | 0.9945 | 0.9786 | 0.9826 |

<span id="page-5-1"></span>Table 3: Performance of TELEClass and its ablations on Amazon-531 and DBPedia-298 datasets. The best score is boldfaced.

| Methods                  |            | Amazon | -531   |        | DBPedia-298 |        |        |        |
|--------------------------|------------|--------|--------|--------|-------------|--------|--------|--------|
| 1120110 40               | Example-F1 | P@1    | P@3    | MRR    | Example-F1  | P@1    | P@3    | MRR    |
| Gen-Only                 | 0.5151     | 0.7477 | 0.5096 | 0.5357 | 0.7930      | 0.9421 | 0.7930 | 0.8209 |
| TELEClass-NoLLMEnrich    | 0.5520     | 0.7370 | 0.5463 | 0.5900 | 0.8319      | 0.9108 | 0.8319 | 0.8563 |
| TELEClass-NoCorpusEnrich | 0.6143     | 0.8358 | 0.6082 | 0.6522 | 0.8185      | 0.8916 | 0.8185 | 0.8463 |
| TELEClass-NoGen          | 0.6449     | 0.8348 | 0.6387 | 0.6792 | 0.8494      | 0.9187 | 0.8494 | 0.8730 |
| TELEClass                | 0.6483     | 0.8505 | 0.6421 | 0.6865 | 0.8633      | 0.9351 | 0.8633 | 0.8864 |

outperforms TaxoClass-NoST, the strongest baseline that, like TELE-Class, does not use self-training. Given that TELEClass uses an even simpler classifier model than TaxoClass-NoST, its superior performance shows the substantially better pseudo training data obtained by combining unlabeled corpus and LLMs. (3) Although LLMs (e.g., ChatGPT) show power in many tasks, naïvely prompting it in the hierarchical text classification task yields significantly inferior performance compared to strong weakly-supervised text classifiers. This proves the necessity of incorporating corpus-based knowledge to improve label taxonomy understanding for the hierarchical setting. We conduct a more detailed comparison with LLM prompting for hierarchical text classification in Section 4.4.

We also study the temporal complexity of TELEClass. We observe that, on Amazon-531, both TELEClass and the strongest baseline TaxoClass take around 5 to 5.5 hours. The reason why TELEClass does not increase the overall temporal complexity is that TaxoClass needs to run the textual entailment model on each pair of document and candidate class. On the other hand, the taxonomy enrichment step of TELEClass makes it possible to simplify this process with embedding similarity calculation which saves a lot of time, while the saved time is budgeted for LLM prompting.

# 4.3 Ablation Studies

We conduct ablation studies to better understand how each component of TELEClass contributes to final performance. Table 3 shows the results of the following ablations:

 Gen-Only only uses the augmented documents by path-based LLM generation to train the final classifier.

- TELEClass-NoLLMEnrich excludes the LLM-based taxonomy enrichment component.
- TELEClass-NoCorpusEnrich excludes the corpus-based taxonomy enrichment component.
- TELEClass-NoGen excludes the augmented documents by pathbased LLM generation.

We find that the full model TELEClass achieves the overall best performance among the compared methods, showing the effectiveness of each of its components. First, both the LLM-based and corpus-based enrichment modules bring improvement to the performance. Interestingly, we find that they make different levels of contribution on the two datasets: LLM-based enrichment brings more improvement on Amazon-531 while corpus-based enrichment contributes more on DBPedia-298. We suspect the reasons are as follows. The classes in Amazon-531 are commonly seen product types that LLM can understand and enrich in a reliable manner. However, DBPedia-298 contains classes that are more subtle to distinguish, which can also be shown by the lower performance of zero-shot LLM prompting on DBPedia compared to Amazon-531 (c.f. ChatGPT in Table 2). Therefore, corpus-based enrichment can consolidate the meaning of each class based on corpus-specific knowledge to facilitate better classification. We also find that path-based LLM generation consistently improves the model performance while requiring only a few hundred queries to LLMs. Even Gen-Only achieves comparable performance to the strong baseline TaxoClass-NoST, demonstrating the effectiveness of this augmentation step.

<span id="page-6-1"></span>Table 4: Performance comparison of TELEClass and zero-shot LLM prompting. We only report Example-F1 and P@k, because it is not straightforward to get ranking of classes predicted by LLMs for MRR calculation. We also report estimated costs in US dollars and running time in minutes for each method on the entire test set. "‡ " indicates that we report the performance based on an estimation from a 1,000-document subset of test data.

| Methods               | Amazon-531 |        |        |           |           | DBPedia-298 |        |        |           |            |
|-----------------------|------------|--------|--------|-----------|-----------|-------------|--------|--------|-----------|------------|
|                       | Example-F1 | P@1    | P@3    | Est. Cost | Est. Time | Example-F1  | P@1    | P@3    | Est. Cost | Est. Time  |
| GPT-3.5-turbo         | 0.5164     | 0.6807 | 0.4752 | \$60      | 240 mins  | 0.4816      | 0.5328 | 0.4547 | \$80      | 400 mins   |
| GPT-3.5-turbo (level) | 0.6621     | 0.8574 | 0.6444 | \$20      | 800 mins  | 0.6649      | 0.8301 | 0.6488 | \$60      | 1,000 mins |
| GPT-4‡                | 0.6994     | 0.8220 | 0.6890 | \$800     | 400 mins  | 0.6054      | 0.6520 | 0.5920 | \$2,500   | 1,000 mins |
| TELEClass             | 0.6483     | 0.8505 | 0.6421 | <\$1      | 3 mins    | 0.8633      | 0.9351 | 0.8633 | <\$1      | 7 mins     |

# <span id="page-6-0"></span>4.4 Comparison with Zero-Shot LLM Prompting

In this section, we further compare TELEClass with zero-shot LLM prompting. Because it is not straightforward to get ranked predictions by LLMs, we only report Example-F1 and P@k as performance evaluation. Additionally, we report the estimated cost and time for each method on the entire test set. The inference time is reported in minutes, and please be aware that this is just a rough estimation as the actual running time is also dependent on the server condition.

We include the following settings to compare with TELEClass:

- GPT-3.5-turbo: We include all the classes in the prompt and ask GPT-3.5-turbo model to provide 3 most appropriate classes for a given document.
- GPT-3.5-turbo (level): We perform level-by-level prompting using GPT-3.5-turbo. Starting from the root node, we ask the model to return one most appropriate class for a given document, and we iteratively prompt the model with the children of the selected node at each level. This method can only generate a path in the taxonomy, but in the actual multi-label hierarchical classification setting, the true labels may not sit in the same path.
- GPT-4: We include all the classes in the prompt and ask GPT-4 to provide 3 most appropriate classes for a given document. Given the limited budget, we only test it on randomly sampled 1,000 documents and estimate the cost on the entire test set.

Table [4](#page-6-1) shows the experiment results. We find that TELEClass consistently outperforms all compared methods on DBPedia, while on Amazon, TELEClass underperforms GPT-3.5-turbo (level) and GPT-4 but still being comparable. As for the cost, once trained, TELEClass does not require additional cost on inference and also has substantially shorter inference time. Prompting LLMs takes longer time and can be prohibitively expensive (e.g., using GPT-4), and the cost will scale up with increasing size of test data. Also, we find that GPT-3.5-turbo (level) consistently outperforms the naïve version, demonstrating the necessity of taxonomy structure. It saves the cost because of the much shorter prompts, but takes longer time due to more queries made per document.

# 4.5 Case studies

To better understand the TELEClass framework, we show some intermediate results of two documents in Table [5,](#page-7-0) including the core classes selected by (1) the original TaxoClass [\[45\]](#page-8-19) method, (2) TELEClass's initial core classes selected by LLM, and (3) refined core classes of TELEClass. Besides, we also include the true labels of the documents and the taxonomy enrichment results of the corresponding core class in the table. Overall, we can see that

TELEClass's refined core class is the most accurate. For example, for the first Wikipedia article about a library, TaxoClass selects "village" as the core class, while TELEClass's initial core class finds a closer one "building" thanks to the power of LLMs. Then, with the enriched classes-indicative features as guidance, TELEClass's refined core class correctly identifies the optimal core class, which is "library". In the other example, TELEClass also pinpoints the most accurate core class "bathroom aids safety" while other methods can only find more general or partially relevant classes.

Besides, although TELEClass outperforms the zero-shot LLM prompting in most cases, there are cases showing the contrary and here is one example we found from Amazon-531. The product review is about "glycolic treatment pads" for which GPT correctly predicts its labels as "beauty" and "skin care", while TELEClass predicts it as "health care". We suspect that the word "treatment" in the review leads to the error because of the bias of term-based pseudo-labeling. It is a known issue of keyword-based methods and some solutions are proposed for the weakly-supervised flat text classification [\[12,](#page-8-31) [65\]](#page-9-3). We hope our study can motivate more research to solve this issue in the hierarchical setting.

# 5 Related Work

# 5.1 Weakly-Supervised Text Classification

Weakly supervised text classification trains a classifier with limited guidance, aiming to reduce human efforts while maintaining high proficiency. Various sources of weak supervision have been explored, including distant supervision [\[8,](#page-8-32) [17,](#page-8-33) [47\]](#page-8-34) like knowledge bases, keywords [\[1,](#page-8-35) [34,](#page-8-36) [36,](#page-8-37) [50,](#page-8-21) [53,](#page-8-25) [62\]](#page-9-4), and heuristic rules [\[3,](#page-8-38) [40,](#page-8-39) [46\]](#page-8-40). Later, extremely weakly-supervised methods are proposed to solely rely on class names to generate pseudo labels and train classifiers. LOTClass [\[36\]](#page-8-37) utilizes MLM-based PLM as a knowledge base for extracting class-indicative keywords. X-Class [\[53\]](#page-8-25) extracts keywords for creating static class representations through clustering. WDDC [\[61\]](#page-9-5) prompts a masked language model and train a classifier over the word distributions. NPPrompt [\[68\]](#page-9-6) uses the PLM embeddings to construct verbalizers and their weighted distribution over each class. PESCO [\[52\]](#page-8-41) performs zero-shot classification with semantic matching between class description and then finetune a classifier with iterative contrastive learning. PIEClass [\[65\]](#page-9-3) employs PLMs' zero-shot prompting to obtain pseudo labels with noise-robust iterative ensemble training. MEGClass [\[26\]](#page-8-42) acquires contextualized sentence representations to capture topical information at the document level. WOTClass [\[51\]](#page-8-43) ranks class-indicative

| Dataset | Document                                                                                                                                                                           | Core Classes by                                                                                                                                                                         | True Labels                                                                   | Corr. Enrichment                                                                                  |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| DBPedia | The Lindenhurst Memorial Library (LML) is<br>located in Lindenhurst, New York, and is one<br>of the fifty six libraries that are part of the<br>Suffolk Cooperative Library System | TaxoClass: village<br>TELEClass initial: building<br>TELEClass refined: library                                                                                                         | library©, agent,<br>educational institution                                   | Class: library<br>Top Enrichment:<br>national library,<br>central library,<br>collection, volumes |
| Amazon  | Since mom (89 yrs young) isn't steady on<br>her feet, we have placed these grab bars<br>around the room. It gives her the stability<br>and security she needs.                     | TaxoClass: personal care,<br>health personal care, safety<br>TELEClass initial: daily living aids,<br>medical supplies equipment, safety,<br>TELEClass refined:<br>bathroom aids safety | health personal care,<br>medical supplies equipment,<br>bathroom aids safety© | Class:<br>bathroom aids safety<br>Top Enrichment:<br>seat, toilet, shower,<br>safety, handles     |

<span id="page-7-0"></span>Table 5: Intermediate results on two documents, including selected core classes by different methods, true labels, and the corresponding taxonomy enrichment results. The optimal core class in the true labels is marked with ©.

keywords from generated classes and extracts classes with overlapping keywords. RulePrompt [\[30\]](#page-8-44) mutually enhances automatically mined logical rules and pseudo labels for fine-tuning a classifier.

# 5.2 Hierarchical Text Classification

A hierarchy provides a systematic top-to-down structure with inherent semantic relations that can assist in text classification. Typical hierarchical text classification can be categorized into two groups: local approaches and global approaches. Local approaches train multiple classifiers for each node or local structures [\[4,](#page-8-45) [31,](#page-8-46) [55\]](#page-8-47). Global approaches, learn hierarchy structure into a single classifier through recursive regularization [\[18\]](#page-8-48), a graph neural network (GNN)-based encoder [\[24,](#page-8-10) [38,](#page-8-49) [45,](#page-8-19) [69\]](#page-9-7), or a joint document label embedding space [\[9\]](#page-8-9). Recent studies also show that LLMs cannot comprehend the complex hierarchical structure [\[6,](#page-8-50) [16,](#page-8-51) [59\]](#page-8-17).

Weak supervision is also studied for hierarchical text classification. WeSHClass [\[35\]](#page-8-18) uses a few keywords or example documents per class and pretrains classifiers with pseudo documents followed by self-training. TaxoClass [\[45\]](#page-8-19) follows the same setting as ours which uses the sole class name of each class as the only supervision. It identifies core classes for each document using a textual entailment model, which is then used to train a multi-label classifier. Additionally, MATCH [\[66\]](#page-9-8) and HiMeCat [\[64\]](#page-9-9) study how to integrate associated metadata into the label hierarchy for document categorization with weak supervision.

# 5.3 LLMs as Generators and Annotators

Large language models (LLMs) have demonstrated impressive performance in many downstream tasks and are explored to help low-resource settings by synthesizing data as generators or annotators [\[11\]](#page-8-52). For data generation, few-shot examples [\[54\]](#page-8-53) or classconditioned prompts [\[33\]](#page-8-54) are explored for LLM generation and the generated data can be used as pseudo training data to further fine-tune a small model as the final classifier [\[56\]](#page-8-55). Recently, Yu et al. [\[60\]](#page-9-10) proposed an attribute-aware topical text classification method that incorporates ChatGPT to generate topic-dependent attributes and topic-independent attributes to reduce topic ambiguity and increase topic diversity for generation. For data annotation, previous works utilize LLMs for unsupervised annotation [\[15\]](#page-8-56), Chain-of-Thought annotation with explanation generation [\[22\]](#page-8-57), and active annotation [\[63\]](#page-9-11).

# 6 Conclusion and Future Work

In this paper, we propose a new method, TELEClass, for the minimallysupervised hierarchical text classification task with two major contributions. First, we enrich the input label taxonomy with LLMgenerated and corpus-based class-indicative terms for each class, which can serve as additional features to understand the classes and facilitate classification. Second, we explore the utilization of LLMs in the hierarchical text classification in two directions: data annotation and data creation. On two public datasets, TELEClass can outperform existing baselines substantially, and we further demonstrate its effectiveness through ablation studies. We also conduct a comparative analysis of performance and cost for zero-shot LLM prompting for the hierarchical text classification task.

For future works, first, we plan to generalize TELEClass's idea of combining LLMs with data-specific knowledge into other lowresource text mining tasks with hierarchical label spaces, such as fine-grained entity typing. Second, in this paper, we mainly focus on acquiring high-quality pseudo labeled data while only utilizing the simplest classifier model and objective. It is worth studying how the proposed method can be further improved with more advanced network structure and noise-robust training objectives. Lastly, we also plan to explore how to extend TELEClassinto harder settings like when existing LLMs do not have the knowledge for the initial annotation (e.g., a private domain), a lower-resource scenario where the availability of the unlabeled corpus is limited, or a more complicated label space like an extremely large hierarchical label space with millions of classes.

# Acknowledgments

Research was supported in part by US DARPA INCAS Program No. HR0011-21-C0165 and BRIES Program No. HR0011-24-3-0325, National Science Foundation IIS-19-56151, the Molecule Maker Lab Institute: An AI Research Institutes program supported by NSF under Award No. 2019897, and the Institute for Geospatial Understanding through an Integrative Discovery Environment (I-GUIDE) by NSF under Award No. 2118329. Any opinions, findings, and conclusions or recommendations expressed herein are those of the authors and do not necessarily represent the views, either expressed or implied, of DARPA or the U.S. Government.

#### References

- <span id="page-8-35"></span>[1] Eugene Agichtein and Luis Gravano. 2000. Snowball: extracting relations from large plain-text collections. In *Digital library*.
- <span id="page-8-15"></span>[2] Anthropic. 2024. Introducing the next generation of Claude. https://www.anthropic.com/news/claude-3-family
- <span id="page-8-38"></span>[3] Sonia Badene, Kate Thompson, Jean-Pierre Lorré, and Nicholas Asher. 2019. Data Programming for Learning Discourse Structure. In ACL.
- <span id="page-8-45"></span>[4] Siddhartha Banerjee, Cem Akkaya, Francisco Perez-Sorrosal, and Kostas Tsioutsiouliklis. 2019. Hierarchical Transfer Learning for Multi-label Text Classification. In ACL.
- <span id="page-8-12"></span>[5] David Berthelot, Nicholas Carlini, Ian Goodfellow, Avital Oliver, Nicolas Papernot, and Colin Raffel. 2019. MixMatch: a holistic approach to semi-supervised learning. In NeurIPS.
- <span id="page-8-50"></span>[6] Rohan Bhambhoria, Lei Chen, and Xiaodan Zhu. 2023. A Simple and Effective Framework for Strict Zero-Shot Hierarchical Classification. In ACL short.
- <span id="page-8-3"></span>[7] Huanhuan Cao, Derek Hao Hu, Dou Shen, Daxin Jiang, Jian-Tao Sun, Enhong Chen, and Qiang Yang. 2009. Context-aware query classification. In SIGIR.
- <span id="page-8-32"></span>[8] Ming-Wei Chang, Lev-Arie Ratinov, Dan Roth, and Vivek Srikumar. 2008. Importance of Semantic Representation: Dataless Classification. In AAAI.
- <span id="page-8-9"></span>[9] Haibin Chen, Qianli Ma, Zhenxi Lin, and Jiangyue Yan. 2021. Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification. In ACL-IJCNLP.
- <span id="page-8-23"></span>[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL-HLT.
- <span id="page-8-52"></span>[11] Bosheng Ding, Chengwei Qin, Linlin Liu, Yew Ken Chia, Boyang Li, Shafiq Joty, and Lidong Bing. 2023. Is GPT-3 a Good Data Annotator?. In ACL.
- <span id="page-8-31"></span>[12] Chengyu Dong, Zihan Wang, and Jingbo Shang. 2023. Debiasing Made State-ofthe-art: Revisiting the Simple Seed-based Weak Supervision for Text Classification. In EMNLP.
- <span id="page-8-0"></span>[13] Susan Dumais and Hao Chen. 2000. Hierarchical classification of Web content. In SIGIR.
- <span id="page-8-6"></span>[14] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130 (2024).
- <span id="page-8-56"></span>[15] Xiachong Feng, Xiaocheng Feng, Libo Qin, Bing Qin, and Ting Liu. 2021. Language Model as an Annotator: Exploring DialoGPT for Dialogue Summarization. In ACL-IJCNLP.
- <span id="page-8-51"></span>[16] Cai Fuhan, Liu Duo, Zhang Zhongqiang, Liu Ge, Yang Xiaozhe, and Fang Xiangzhong. 2024. NER-guided Comprehensive Hierarchy-aware Prompt Tuning for Hierarchical Text Classification. In *LREC-COLING*.
- <span id="page-8-33"></span>[17] Evgeniy Gabrilovich and Shaul Markovitch. 2007. Computing Semantic Relatedness Using Wikipedia-Based Explicit Semantic Analysis. In IJCAI.
- <span id="page-8-48"></span>[18] Siddharth Gopal and Yiming Yang. 2013. Recursive regularization for large-scale classification with hierarchical and graphical dependencies. In KDD.
- <span id="page-8-26"></span>[19] Xiaotao Gu, Zihan Wang, Zhenyu Bi, Yu Meng, Liyuan Liu, Jiawei Han, and Jingbo Shang. 2021. UCPhrase: Unsupervised Context-aware Quality Phrase Tagging. In KDD.
- <span id="page-8-13"></span>[20] Suchin Gururangan, Tam Dang, Dallas Card, and Noah A. Smith. 2019. Variational Pretraining for Semi-supervised Text Classification. In ACL.
- <span id="page-8-4"></span>[21] Bing He, Sreyashi Nag, Limeng Cui, Suhang Wang, Zheng Li, Rahul Goutam, Zhen Li, and Haiyang Zhang. 2024. Hierarchical Query Classification in E-commerce Search. In WWW Companion.
- <span id="page-8-57"></span>[22] Xingwei He, Zheng-Wen Lin, Yeyun Gong, Alex Jin, Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan Duan, and Weizhu Chen. 2024. AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators. In NAACL.
- <span id="page-8-5"></span>[23] Yunzhong He, Cong Zhang, Ruoyan Kong, Chaitanya Kulkarni, Qing Liu, Ashish Gandhe, Amit Nithianandan, and Arul Prakash. 2023. HierCat: Hierarchical Query Categorization from Weakly Supervised Data at Facebook Marketplace. In WWW Companion.
- <span id="page-8-10"></span>[24] Wei Huang, Enhong Chen, Qi Liu, Yuying Chen, Zai Huang, Yang Liu, Zhou Zhao, Dan Zhang, and Shijin Wang. 2019. Hierarchical Multi-label Text Classification: An Attention-based Recurrent Network Approach. In CIKM.
- <span id="page-8-1"></span>[25] SeongKu Kang, Shivam Agarwal, Bowen Jin, Dongha Lee, Hwanjo Yu, and Jiawei Han. 2024. Improving Retrieval in Theme-specific Applications using a Corpus Topical Taxonomy. In WWW.
- <span id="page-8-42"></span>[26] Priyanka Kargupta, Tanay Komarlu, Susik Yoon, Xuan Wang, and Jiawei Han. 2023. MEGClass: Extremely Weakly Supervised Text Classification via Mutually-Enhancing Text Granularities. In Findings of EMNLP.
- <span id="page-8-30"></span>[27] Quoc Le and Tomas Mikolov. 2014. Distributed Representations of Sentences and Documents. In *ICML*.
- <span id="page-8-28"></span>[28] Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, S. Auer, and Christian Bizer. 2015. DBpedia - A large-scale, multilingual knowledge base extracted from Wikipedia. Semantic Web 6 (2015), 167–195.
- <span id="page-8-2"></span>base extracted from Wikipedia. Semantic Web 6 (2015), 167–195.
  [29] Keqian Li, Shiyang Li, Semih Yavuz, Hanwen Zha, Yu Su, and Xifeng Yan. 2019.
  HierCon: Hierarchical Organization of Technical Documents Based on Concepts.

- In ICDM.
- <span id="page-8-44"></span>[30] Miaomiao Li, Jiaqi Zhu, Yang Wang, Yi Yang, Yilin Li, and Hongan Wang. 2024. RulePrompt: Weakly Supervised Text Classification with Prompting PLMs and Self-Iterative Logical Rules. In WWW.
- <span id="page-8-46"></span>[31] Tieyan Liu, Yiming Yang, Hao Wan, Huajun Zeng, Zheng Chen, and Weiying Ma. 2005. Support vector machines classification with a very large-scale taxonomy. In KDD.
- <span id="page-8-27"></span>[32] Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In RecSys.
- <span id="page-8-54"></span>[33] Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Generating Training Data with Language Models: Towards Zero-Shot Language Understanding. In NeurIPS.
- <span id="page-8-36"></span>[34] Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2018. Weakly-Supervised Neural Text Classification. In CIKM.
- <span id="page-8-18"></span>[35] Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2019. Weakly-supervised hierarchical text classification. In AAAI.
- <span id="page-8-37"></span>[36] Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang, and Jiawei Han. 2020. Text Classification Using Label Names Only: A Language Model Self-Training Approach. In EMNLP.
- <span id="page-8-14"></span>[37] OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 (2023).
- <span id="page-8-49"></span>[38] Hao Peng, Jianxin Li, Yu He, Yaopeng Liu, Mengjiao Bao, Yangqiu Song, and Qiang Yang. 2018. Large-Scale Hierarchical Text Classification with Recursively Regularized Deep Graph-CNN. In WWW.
- <span id="page-8-8"></span>[39] Mykhailo Poliakov and Nadiya Shvai. 2024. Multi-Meta-RAG: Improving RAG for Multi-Hop Queries using Database Filtering with LLM-Extracted Metadata. arXiv preprint arXiv:2406.13213 (2024).
- <span id="page-8-39"></span>[40] Alexander Ratner, Christopher De Sa, Sen Wu, Daniel Selsam, and Christopher Ré. 2016. Data Programming: Creating Large Training Sets, Quickly. In NIPS.
- <span id="page-8-20"></span>[41] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In EMNLP.
- <span id="page-8-22"></span>[42] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (apr 2009), 333–389.
- <span id="page-8-7"></span>[43] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In ICLR.
- <span id="page-8-24"></span>[44] Jingbo Shang, Jialu Liu, Meng Jiang, Xiang Ren, Clare R. Voss, and Jiawei Han. 2018. Automated Phrase Mining from Massive Text Corpora. TKDE 30, 10 (2018), 1825–1837.
- <span id="page-8-19"></span>[45] Jiaming Shen, Wenda Qiu, Yu Meng, Jingbo Shang, Xiang Ren, and Jiawei Han. 2021. TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names. In NAACL.
- <span id="page-8-40"></span>[46] Kai Shu, Subhabrata Mukherjee, Guoqing Zheng, Ahmed Hassan Awadallah, Milad Shokouhi, and Susan Dumais. 2020. Learning with Weak Supervision for Email Intent Detection. In SIGIR.
- <span id="page-8-34"></span>[47] Yangqiu Song and Dan Roth. 2014. On Dataless Hierarchical Text Classification. In AAAI.
- <span id="page-8-58"></span>[48] T. Sørensen. 1948. A Method of Establishing Groups of Equal Amplitude in Plant Sociology Based on Similarity of Species Content and Its Application to Analyses of the Vegetation on Danish Commons. Munksgaard in Komm.
- <span id="page-8-16"></span>[49] Xiaofei Sun, Xiaoya Li, Jiwei Li, Fei Wu, Shangwei Guo, Tianwei Zhang, and Guoyin Wang. 2023. Text Classification via Large Language Models. In Findings of EMNLP.
- <span id="page-8-21"></span>[50] Fangbo Tao, Chao Zhang, Xiusi Chen, Meng Jiang, Tim Hanratty, Lance Kaplan, and Jiawei Han. 2018. Doc2Cube: Allocating Documents to Text Cube Without Labeled Data. In *ICDM*.
- <span id="page-8-43"></span>[51] Tianle Wang, Zihan Wang, Weitang Liu, and Jingbo Shang. 2023. WOT-Class: Weakly Supervised Open-world Text Classification. In CIKM.
- <span id="page-8-41"></span>[52] Yau-Shian Wang, Ta-Chung Chi, Ruohong Zhang, and Yiming Yang. 2023. PESCO: Prompt-enhanced Self Contrastive Learning for Zero-shot Text Classification. In ACL.
- <span id="page-8-25"></span>[53] Zihan Wang, Dheeraj Mekala, and Jingbo Shang. 2021. X-Class: Text Classification with Extremely Weak Supervision. In NAACL.
- <span id="page-8-53"></span>[54] Zirui Wang, Adams Wei Yu, Orhan Firat, and Yuan Cao. 2021. Towards Zero-Label Language Learning. arXiv preprint arXiv:2109.09193 (2021).
- <span id="page-8-47"></span>[55] Jonatas Wehrmann, Ricardo Cerri, and Rodrigo C. Barros. 2018. Hierarchical Multi-label Classification Networks. In ICML.
- <span id="page-8-55"></span>[56] Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong. 2022. ZeroGen: Efficient Zero-shot Learning via Dataset Generation. In EMNLP.
- <span id="page-8-29"></span>[57] Wenpeng Yin, Jamaal Hay, and Dan Roth. 2019. Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. In EMNLP-IJCNLP.
- <span id="page-8-11"></span>[58] Ronghui You, Zihan Zhang, Ziye Wang, Suyang Dai, Hiroshi Mamitsuka, and Shanfeng Zhu. 2019. AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification. In NeurIPS.
- <span id="page-8-17"></span>[59] Simon Chi Lok Yu, Jie He, Victor Basulto, and Jeff Pan. 2023. Instances and Labels: Hierarchy-aware Joint Supervised Contrastive Learning for Hierarchical Multi-Label Text Classification. In Findings of EMNLP.

#### **Algorithm 1:** TELEClass

<span id="page-9-2"></span>**Input:** A corpus  $\mathcal{D}$ , a label taxonomy  $\mathcal{T}$ , a pretrained text encoder  $\mathcal{S}$ , an LLM  $\mathcal{G}$ .

**Output:** A text classifier F that can classify each document into a set of classes in  $\mathcal{T}$ .

- 1 // LLM-Enhanced Core Class Annotation;
- 2 for  $c \in C$  do
- $T_c^{\text{LLM}} \leftarrow \text{use } G \text{ to enrich } c \text{ with key terms};$
- 4 for  $d_i \in \mathcal{D}$  do
- $\mathbb{C}_{i}^{0} \leftarrow \text{use } \mathcal{G} \text{ to select core classes from candidates}$  retrieved using  $\mathcal{S}$  and  $T_{c}^{\text{LLM}}$ ;
- 6 // Corpus-Based Taxonomy Enrichment;
- 7 for  $c \in C$  do
- $D_c^0 \leftarrow$  a set of roughly classified documents;
- $for c_p \in Par(c) do$ 
  - $T(c, c_p) \leftarrow \text{top terms ranked by affinity};$
- 11  $T_c \leftarrow$  aggregate corpus-based and LLM-generated terms Eq. 4;
- 12 // Core Class Refinement with Enriched Taxonomy;
- 13  $\vec{d} \leftarrow$  document representation S(d);
- 14 for  $c \in C$  do
- 15  $D_c \leftarrow$  confident documents by matching  $T_c$ ;
- $\vec{c} \leftarrow$  average document representation in  $D_c$ ;
- 17 for  $d_i \in \mathcal{D}$  do
- $\mathbb{C}_i, conf_i \leftarrow \text{refined core classes using } \cos(\vec{d}, \vec{c}) \text{ and Eq.}$  5;
- 19  $\mathcal{D}^{core} \leftarrow$  confident refined core classes;
- 20 // Text Classifier Training with Path-Based Data Augmentation;
- 21  $\mathcal{D}^{\text{gen}} \leftarrow \text{generate } q \text{ documents for each path using } \mathcal{G};$
- 22  $F \leftarrow$  train classifier with  $\mathcal{D}^{core}$  and  $\mathcal{D}^{gen}$  Eq. 6;
- 23 Return F;
- <span id="page-9-10"></span>[60] Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, and Chao Zhang. 2023. Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias. In NeurIPS.
- <span id="page-9-5"></span>[61] Ziqian Zeng, Weimin Ni, Tianqing Fang, Xiang Li, Xinran Zhao, and Yangqiu Song. 2022. Weakly Supervised Text Classification using Supervision Signals from a Language Model. In Findings of NAACL.
- <span id="page-9-4"></span>[62] Lu Zhang, Jiandong Ding, Yi Xu, Yingyao Liu, and Shuigeng Zhou. 2021. Weakly-supervised Text Classification Based on Keyword Graph. In EMNLP.
- <span id="page-9-11"></span>[63] Ruoyu Zhang, Yanzeng Li, Yongliang Ma, Ming Zhou, and Lei Zou. 2023. LLMaAA: Making Large Language Models as Active Annotators. In Findings of EMNLP.
- <span id="page-9-9"></span>[64] Yu Zhang, Xiusi Chen, Yu Meng, and Jiawei Han. 2021. Hierarchical Metadata-Aware Document Categorization under Weak Supervision. In WSDM.
- <span id="page-9-3"></span>[65] Yunyi Zhang, Minhao Jiang, Yu Meng, Yu Zhang, and Jiawei Han. 2023. PIEClass: Weakly-Supervised Text Classification with Prompting and Noise-Robust Iterative Ensemble Training. In EMNLP.
- <span id="page-9-8"></span>[66] Yu Zhang, Zhihong Shen, Yuxiao Dong, Kuansan Wang, and Jiawei Han. 2021. MATCH: Metadata-Aware Text Classification in A Large Hierarchy. In WWW.
- <span id="page-9-0"></span>[67] Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, and Jiawei Han. 2023. Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts. In WSDM.
- <span id="page-9-6"></span>[68] Xuandong Zhao, Siqi Ouyang, Zhiguo Yu, Ming Wu, and Lei Li. 2023. Pre-trained Language Models Can be Fully Zero-Shot Learners. In ACL.
- <span id="page-9-7"></span>[69] Jie Zhou, Chunping Ma, Dingkun Long, Guangwei Xu, Ning Ding, Haoyu Zhang, Pengjun Xie, and Gongshen Liu. 2020. Hierarchy-Aware Global Model for Hierarchical Text Classification. In ACL.

# A Pseudo code of TELEClass

Algorithm 1 summarizes the TELEClass method.

# <span id="page-9-1"></span>**B** Prompts for LLM

#### • LLM-based enrichment for Amazon-531

**Instruction:** [Target Class] is a product class in Amazon and is the subclass of [Parent Class]. Please generate 10 additional key terms about the [Target Class] that are relevant to [Target Class] but irrelevant to [Sibling Classes]. Please split the additional key terms using commas.

#### • LLM-based enrichment for DBPedia-298

**Instruction:** [Target Class] is an article category of Wikipedia articles and is the subclass of [Parent Class]. Please generate 10 additional key terms about the [Target Class] that are relevant to [Target Class] but irrelevant to [Sibling Classes]. Please split the additional key terms using commas.

#### • Core class annotation for Amazon-531

**Instruction:** You will be provided with an Amazon product review, and please select its product types from the following categories: [Candidate Classes]. Just give the category names as shown in the provided list.

Query: [Document]

#### • Core class annotation for DBPedia-298

**Instruction:** You will be provided with a Wikipedia article describing an entity at the beginning, and please select its types from the following categories: [Candidate Classes]. Just give the category names as shown in the provided list. **Query:** [Document]

# • Path-based generation for Amazon-531

**Instruction:** Suppose you are an Amazon Reviewer, please generate 5 various and reliable passages following the requirements below:

- 1. Must generate reviews following the themes of the taxonomy path: [Path].
- 2. Must be in length about 100 words.
- The writing style and format of the text should be a product review.
- 4. Should keep the generated text to be diverse, specific, and consistent with the given taxonomy path. You should focus on [The Leaf Node on the Path].

#### • Path-based generation for DBPedia-298

**Instruction:** Suppose you are a Wikipedia Contributor, please generate 5 various and reliable passages following the requirements below:

- 1. Must generate reviews following the themes of the taxonomy path: [Path].
- 2. Must be in length about 100 words.
- 3. The writing style and format of the text should be a Wikipedia page.
- 4. Should keep the generated text to be diverse, specific, and consistent with the given taxonomy path. You should focus on [The Leaf Node on the Path].

# <span id="page-10-1"></span><span id="page-10-0"></span>**C** Evaluation Metrics

Let  $\mathbb{C}_i^{\text{true}}$  and  $\mathbb{C}_i^{\text{pred}}$  denote the set of true labels and the set of predicted labels for document  $d_i \in \mathcal{D}$ , respectively. If a method can generate rankings of classes within  $\mathbb{C}_i^{\text{pred}}$ , we further denote its top-k predicted labels as  $\mathbb{C}_{i,k}^{\text{pred}} \subset \mathbb{C}_i^{\text{pred}}$ . The evaluation metrics are defined as follows:

 Example-F1 [48], which is also called micro-Dice coefficient, evaluates the multi-label classification results without ranking,

Example-F1 = 
$$\frac{1}{|\mathcal{D}|} \sum_{d_i \in \mathcal{D}} \frac{2 \cdot |\mathbb{C}_i^{\text{true}} \cap \mathbb{C}_i^{\text{pred}}|}{|\mathbb{C}_i^{\text{true}}| + |\mathbb{C}_i^{\text{pred}}|}.$$
 (7)

 Precision at k, or P@k, is a ranking-based metric that evaluates the precision of top-k predicted classes,

$$P@k = \frac{1}{k} \sum_{d_i \in \mathcal{D}} \frac{|\mathbb{C}_i^{\text{true}} \cap \mathbb{C}_{i,k}^{\text{pred}}|}{\min(k, |\mathbb{C}_i^{\text{true}}|)}.$$
 (8)

 Mean Reciprocal Rank, or MRR, is another ranking-based metric, which evaluates the multi-label predictions based on the inverse of true labels' ranks within predicted classes,

$$MRR = \frac{1}{|\mathcal{D}|} \sum_{d_i \in \mathcal{D}} \frac{1}{|\mathbb{C}_i^{true}|} \sum_{c_j \in \mathbb{C}_i^{true}} \frac{1}{\min\{k | c_j \in \mathbb{C}_{i,k}^{pred}\}}.$$
 (9)
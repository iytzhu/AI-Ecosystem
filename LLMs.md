# Large Language models timeline and key papers
[Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) & [Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
## 1. Foundational Theory and Early Pioneering Works
- **2017 arXiv(NeurIPS 2020): “Attention Is All You Need” (Vaswani et al.)**

  > [Paper](https://arxiv.org/abs/1706.03762) & [Openreview](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Reviews.html) & [Code(Original Tensorflow version)](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) & [Code(Pytorch version)](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

## 2. Core Large Language models
- **2020 arXiv(NeurIPS 2020): “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” (Lewis et al.)** (RAG)

  > [Paper](https://arxiv.org/abs/2005.11401) & [Video](https://www.youtube.com/watch?v=JGpmQvlYRdU) & [Blog](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)
  
  An LLM’s parameters essentially represent the general patterns of how humans use words to form sentences. Parameterized knowledge makes LLMs useful in responding to general prompts. However, it doesn’t serve users who want a deeper dive into a specific type of information. RAG can link generative AI services to external resources, especially ones rich in the latest technical details. It is a “general-purpose fine-tuning recipe”, because it can be used by nearly any LLM to connect with practically any external resource.

  Given the prompt “When did the first mammal appear on Earth?” for instance, RAG might surface documents for “Mammal,” “History of Earth,” and “Evolution of Mammals.” These supporting documents are then concatenated as context with the original input and fed to the seq2seq model that produces the actual output. RAG thus has two sources of knowledge: the knowledge that seq2seq models store in their parameters (parametric memory) and the knowledge stored in the corpus from which RAG retrieves passages (nonparametric memory).

  [Here](https://video-lhr6-2.xx.fbcdn.net/o1/v/t2/f2/m69/AQNwycrlbizagrgj0JHtQnpQh_TG8YyE91LMk9pTlC8_Dt2U-S_tEi_joU9-uE6mw8JtChnUTPhpjcWIOtJUEf7r.mp4?strext=1&_nc_cat=104&_nc_sid=8bf8fe&_nc_ht=video-lhr6-2.xx.fbcdn.net&_nc_ohc=VSsV-5RWr28Q7kNvwEj-Cwt&efg=eyJ2ZW5jb2RlX3RhZyI6Inhwdl9wcm9ncmVzc2l2ZS5GQUNFQk9PSy4uQzMuNjQwLnN2ZV9zZCIsInhwdl9hc3NldF9pZCI6MzE2NTUxNjI3OTIwMjAwLCJhc3NldF9hZ2VfZGF5cyI6MTc4MCwidmlfdXNlY2FzZV9pZCI6MTAxMjgsImR1cmF0aW9uX3MiOjM0LCJ1cmxnZW5fc291cmNlIjoid3d3In0%3D&ccb=17-1&_nc_gid=iT6hU-9MhFEYf5KXPjxQhQ&_nc_zt=28&oh=00_AfUM-TyCD0GjjBOGYCjFY_5JD2QsO9ikZAlsnkNdSJ9dNw&oe=689DF056&bitrate=29198&tag=sve_sd) is a demo.

  RAG gives models sources they can cite, like footnotes in a research paper, so users can check any claims. It also reduces the possibility that a model will give a very plausible but incorrect answer, a phenomenon called hallucination.

- **2021 arXiv(ICLR 2022): “LoRA: Low-Rank Adaptation of Large Language Models” (J. Hu et al.)**

  > [Paper](https://arxiv.org/abs/2106.09685) & [OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9) & [Video](https://www.youtube.com/watch?v=DhRoTONcyZE) & [Code](https://github.com/microsoft/LoRA)

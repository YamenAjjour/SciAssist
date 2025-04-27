# SciAssist

This is a research assistants for NLP Researchers that is implemented using Retrieval Augmented Generation. It uses
DeepSeek-R1-Distill-Qwen-1.5B as a backbone to answer questions on scientific NLP Papers. The paper pool that the underlines 
the rag system contains 72k papers from the [ACL Anthology](https://github.com/shauryr/ACL-anthology-corpus?tab=readme-ov-file).

# Deployment


### Locally using CLI
To deploy the Retrieval Augmented Generation pipeline via clli which allows you to chat with the interface via command line you 
can run the following

```
python scirag/setup_rag.py --debug --path-index data/index --path-dataset data/acl-publication-info.74k.parquet
  --path-model /bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B
```

adding debug will create an index of only 100 papers for debugging purposes

### Locally as a web service


fastapi run scirag/setup_api.py --port 80

### Locally as a webservice using docker
```
./sciarg/deploy_sciarg.sh
```

The web service can be then accessed as follows 
```
http://127.0.0.1:8000/chat?q= What are typical software designs of RAG   
```

or using curl 
```
curl curl http://127.0.0.1:80/chat?q=aaa
```

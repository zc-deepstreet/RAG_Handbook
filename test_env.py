# 1. 验证 pyarrow
import pyarrow
print("pyarrow 导入成功")

# 2. 验证 sklearn
import sklearn
print("sklearn 导入成功")

# 3. 验证 sentence_transformers
import sentence_transformers
print("sentence_transformers 导入成功")

# 4. 验证 langchain_huggingface Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("HuggingFaceEmbeddings 初始化成功")
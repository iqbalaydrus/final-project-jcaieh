# find_llmchain.py
print("Attempting to find the location of LLMChain...")
found = False
try:
    from langchain.chains import LLMChain
    print("FOUND: from langchain.chains import LLMChain")
    found = True
except Exception as e:
    print(f"FAILED: from langchain.chains import LLMChain -> {e}")

try:
    from langchain_community.chains import LLMChain
    print("FOUND: from langchain_community.chains import LLMChain")
    found = True
except Exception as e:
    print(f"FAILED: from langchain_community.chains import LLMChain -> {e}")

try:
    from langchain_core.chains import LLMChain
    print("FOUND: from langchain_core.chains import LLMChain")
    found = True
except Exception as e:
    print(f"FAILED: from langchain_core.chains import LLMChain -> {e}")

try:
    from langchain import chains
    print("FOUND: from langchain import chains")
    found = True
except Exception as e:
    print(f"FAILED: from langchain import chains -> {e}")

if not found:
    print("\nCould not find LLMChain in common locations.")

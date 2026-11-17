from dotenv import load_dotenv

load_dotenv(".env")

from langchain_ollama import ChatOllama
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from cache import SemanticCache, ask

langfuse = get_client()
llm = ChatOllama(model="llama3.2", temperature=0, num_predict=80)
cache = SemanticCache()

questions = [
    "Why is etcdNoLeader firing?",
    "What causes the etcdNoLeader alert?",
    "How do I fix KubePodCrashLooping?",
    "How can I fix KubePodCrashLooping?",
    "What does KubeNodeNotReady mean?",
    "What is the meaning of KubeNodeNotReady?",
    "Why is KubeDeploymentReplicasMismatch firing?",
    "What causes the KubeDeploymentReplicasMismatch alert?",
    "How do I resolve KubePersistentVolumeFillingUp?",
    "How can I resolve KubePersistentVolumeFillingUp?",
    "Why is etcdMembersDown firing?",
    "What causes the etcdMembersDown alert?",
    "How do I fix KubeCPUOvercommit?",
    "What steps fix KubeCPUOvercommit?",
    "What should I check first when NodeFilesystemSpaceFillingUp fires?",
    "NodeFilesystemSpaceFillingUp just paged me, where do I start?",
    "Is KubeJobFailed something I need to act on right away?",
    "How urgent is a KubeJobFailed alert?",
    "Why would TargetDown fire for a healthy service?",
    "TargetDown is alerting but the app looks fine, what's going on?",
]
for question in questions:
    with langfuse.start_as_current_observation(name="question", input=question) as trace:
        answer, hit = ask(question, cache, lambda q: (llm.invoke(q, config={"callbacks": [CallbackHandler()]}).content, []))
        trace.update(output=answer, metadata={"hit": hit})

langfuse.flush()


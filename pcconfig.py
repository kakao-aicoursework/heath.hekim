import pynecone as pc

class PyneconeframeConfig(pc.Config):
    pass

config = PyneconeframeConfig(
    app_name="chatbot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)
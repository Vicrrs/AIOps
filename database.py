import sqlite3
import bcrypt
from contextlib import contextmanager

DATABASE_NAME = "aiops.db"

@contextmanager
def get_db():
    """Context manager pra conexao com banco"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # vai retornar os resultados como dicionarios
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
        
def init_db():
    """Inicializa o banco de dados com as tabelas necessarias"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Tabela de usuários
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
        """)

        # Tabela de API Keys
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        key_hash TEXT NOT NULL,
                        key_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        last_used_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
        """)
        
        # Tabelas de modelos permitidos por usuario
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        UNIQUE(user_id, model_name)
                    )
        """)
        
        # indices de performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_models_user ON user_models(user_id)")

def hash_api_key(api_key: str) -> str:
    """Cria Hash da API key usando bcrypt"""
    return bcrypt.checkpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
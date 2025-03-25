# import subprocess
import sqlite3

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

# def create_table_filipino(conn):
#     try:
#         create_table = """
#         CREATE TABLE IF NOT EXISTS filipino (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             word TEXT NOT NULL UNIQUE
#         );
#         """
#         cursor = conn.cursor()
#         cursor.execute(create_table)
#     except sqlite3.Error as e:
#         print(e)


# def create_table_foreign(conn):
#     try:
#         create_table = """
#         CREATE TABLE IF NOT EXISTS overseas (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             word TEXT NOT NULL UNIQUE
#         );
#         """
#         cursor = conn.cursor()
#         cursor.execute(create_table)
#     except sqlite3.Error as e:
#         print(e)

# def insert_data(conn, word, table):
#     try:
#         sql_insert = f"INSERT INTO {table} (word) VALUES (?)"
#         cursor = conn.cursor()
#         cursor.execute(sql_insert, (word,))
#         conn.commit()
#         # print("User inserted with ID:", cursor.lastrowid)
#     except sqlite3.IntegrityError:
#         pass
#         # print(f"'{word}' already exists in the database")

# def query_data(conn):
#     sql_query = "SELECT * FROM filipino"
#     cursor = conn.cursor()
#     cursor.execute(sql_query)
#     rows = cursor.fetchall()
#     for row in rows:
#         print(row)

def word_exists(conn, word):
    sql_query = f"SELECT 1 FROM 'entries' WHERE word = ? LIMIT 1"
    cursor = conn.cursor()
    cursor.execute(sql_query, (word,))
    return cursor.fetchone() is not None


# def stem_filipino(words):
#     """Call Haskell script to stem Filipino words"""
    
# import dictdatabase as DDB

# DDB.config.storage_directory = "./known_words"
# DDB.config.use_compression = False
# DDB.config.indent = None
# DDB.config.use_orjson = True

def identify(word:str):
    database = "../english_dictionary.db"
    conn = create_connection(database)
    if word_exists(conn, f"{word}"):
        return True
    else:
        return False
    # if DDB.at("known_words").exists():
        
        # create_table_filipino(conn)
        # create_table_foreign(conn)
        # exists_in_filipino = word_exists(conn, f"{word}", "filipino")
        # exists_in_foreign = word_exists(conn, f"{word}", "overseas")
        # if exists_in_foreign:
        #     return True
        # if exists_in_filipino:
        #     return False
        # else:
        #     command = f"ghc stemmer.hs {word}"
        #     try:
        #         result = subprocess.run(command, capture_output=True, text=True, check=True)
        #         return result.stdout.strip()
        #     except subprocess.CalledProcessError as e:
        #         print("Error:", e)
        #         return None
        #     return False
        # if DDB.at("known_words", key=word).exists():
        #     return False
        # else:
        #     inp = input(f"Is \"{word}\" foreign? type 'y' for yes or hit enter ")
        #     if inp == "y":
        #         # insert_data(conn, f"{word}", "overseas")
        #         # conn.close()
        #         return True
        #     else:
        #         # insert_data(conn, f"{word}", "filipino")
        #         # conn.close()
        #         return False
    # else:
    #     print("Error! Cannot create the database connection.")

import csv
import psycopg2
import credentials
import pandas as pd
from sqlalchemy import create_engine

conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database="yiqizhou",
                        user=credentials.user_name, password=credentials.pass_word)

print("connected!")

cur = conn.cursor()

cur.execute("DROP TABLE articles CASCADE")

cur.execute("""
    create table articles(
        TextID varchar(10) primary key, semanticobjscore integer, CC integer, CD integer, EX integer, FW integer, 
        INs integer, LS integer, MD integer, NN integer, NNPS integer, POS integer, PRP integer, PRP$ integer, 
        SYM integer, UH integer, VB integer, VBD integer, VBG integer, VBN integer, VBP integer, WDT integer, 
        WP$ integer, baseform integer, Quotes integer, fullstops integer, 
        compsupadjadv integer, past integer, imperative integer, present3rd integer,
        txtcomplexity integer, Outcome integer);
""")

with open('sports_articles.csv', 'r', encoding="utf8") as f:
    copy_sql = """
                   COPY articles FROM stdin WITH CSV HEADER
                   DELIMITER as ','
                   """
    cur.copy_expert(sql=copy_sql, file=f)
    conn.commit()

print("articles inserted!")

# cur.execute("SELECT * INTO tmp_table FROM articles")
# cur.execute("alter table tmp_table drop column outcome")
# cur.execute("alter table tmp_table drop column textid")
# cur.execute("select * from tmp_table")
cur.execute("select * from articles;")
result = cur.fetchall()
#print(result)
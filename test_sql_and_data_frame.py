import psycopg2
from Tree import *
import credentials

def load_data(file_name):
    with open(str(file_name)) as f:
        file = f.readlines()
        label = file[0].strip().split(",")[1:-1]
        file = file[1:]
        row = len(file)
        col = len(file[0].strip().split(","))-1
        mat = np.zeros((row, col))
        for i in range(len(file)):
            row = file[i].strip()
            cols = row.split(",")[1:]
            for j in range(len(cols)):
                mat[i][j] = int(cols[j])
    np.random.seed(10)
    np.random.shuffle(mat)
    X = mat[:, 0:-1]
    y = mat[:, -1]

    return X, y, label

def create_table():
    conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database="yiqizhou",
                            user=credentials.user_name, password=credentials.pass_word)
    print("connected!")
    cur = conn.cursor()
    cur.execute("DROP TABLE small_train, small_test CASCADE")
    cur.execute("""
        create table small_train(
            ID varchar(10) primary key, 
            x1 integer, 
            x2 integer, 
            x3 integer, 
            x4 integer, 
            x5 integer, 
            Outcome integer);
    """)

    cur.execute("""
        create table small_test(
            ID varchar(10) primary key, 
            x1 integer, 
            x2 integer, 
            x3 integer, 
            x4 integer, 
            x5 integer, 
            Outcome integer);
    """)

    with open('small_train.csv', 'r', encoding="utf8") as f:
        copy_sql = """
                       COPY small_train FROM stdin WITH CSV HEADER
                       DELIMITER as ','
                       """
        cur.copy_expert(sql=copy_sql, file=f)
        conn.commit()

    print("small dataset inserted!")

    with open('small_test.csv', 'r', encoding="utf8") as f:
        copy_sql = """
                       COPY small_test FROM stdin WITH CSV HEADER
                       DELIMITER as ','
                       """
        cur.copy_expert(sql=copy_sql, file=f)
        conn.commit()

def test_split_var(train_tree, SQL_train_tree):
    assert train_tree.split_var == SQL_train_tree.split_var

def test_split_point(train_tree, SQL_train_tree):
    assert train_tree.label == SQL_train_tree.label

def test_child(train_tree, SQL_train_tree):
    assert train_tree.left.label == SQL_train_tree.left.label
    assert train_tree.right.label == SQL_train_tree.right.label

def test_all(train_tree, SQL_train_tree):
    test_split_var(train_tree, SQL_train_tree)
    test_split_point(train_tree, SQL_train_tree)
    test_child(train_tree, SQL_train_tree)


if __name__ == "__main__":
    """
    implement unit tests to compare the results of classifier on SQL version and the data frame version
    """
    X_train, y_train, labels = load_data("small_train.csv")
    X_test, y_test, labels = load_data("small_test.csv")
    criterion = 3  # gini_index
    #create_table()
    conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database="yiqizhou", user="yiqizhou", password="19960129zyq")
    cur = conn.cursor()
    print("connected!")
    cur.execute(open("Resources//impurity.sql", "r").read())
    print("opened!")

    train = Tree(criterion, X_train, y_train, labels)
    test = Tree(criterion, X_test, y_test, labels)
    train_tree = train.build_tree(train.root, train.X, train.y, 0, 5, train.labels, train.root.X_idx, None)
    train.print_tree(train_tree)
    pruned_tree, pruned_alpha, error = train.prune_tree(train_tree, 3, test.X, test.y, test.labels)

    SQL_train = SQLTree(criterion, X_train, y_train, labels, "small_train", cur)
    SQL_test = SQLTree(criterion, X_test, y_test, labels, "small_test", cur)
    SQL_train_tree = SQL_train.build_tree(SQL_train.root, 0, 5, SQL_train.labels, None, "")
    SQL_train.print_tree(SQL_train_tree)
    pruned_SQLtree, pruned_alpha, error = SQL_train.prune_tree(SQL_train_tree, 3, SQL_test.X, SQL_test.y, SQL_test.labels)

    test_all(train_tree, SQL_train_tree)

    cur.close()
    print("closed!")
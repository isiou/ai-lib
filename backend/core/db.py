import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = "data/library.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            topic TEXT NOT NULL,
            call_number TEXT NOT NULL,
            location TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    # Check if data exists
    cursor.execute("SELECT COUNT(*) FROM books")
    if cursor.fetchone()[0] == 0:
        logger.info("Initializing test data for library...")
        test_data = [
            # 计算机与人工智能
            (
                "深度学习 (Deep Learning)",
                "Ian Goodfellow",
                "人工智能",
                "TP311.56/123",
                "二楼自然科学阅览室",
                "在馆可借",
            ),
            (
                "人工智能：一种现代的方法",
                "Stuart Russell",
                "人工智能",
                "TP311.56/124",
                "二楼自然科学阅览室",
                "已借出",
            ),
            (
                "Python编程：从入门到实践",
                "Eric Matthes",
                "Python",
                "TP311.56/125",
                "三楼计算机阅览室",
                "在馆可借",
            ),
            (
                "算法导论 (Introduction to Algorithms)",
                "Thomas H. Cormen",
                "计算机科学",
                "TP311.12/45",
                "三楼计算机阅览室",
                "在馆可借",
            ),
            (
                "代码整洁之道 (Clean Code)",
                "Robert C. Martin",
                "软件工程",
                "TP311.5/88",
                "三楼计算机阅览室",
                "在馆可借",
            ),
            (
                "设计模式：可复用面向对象软件的基础",
                "Erich Gamma",
                "软件工程",
                "TP311.5/92",
                "三楼计算机阅览室",
                "已借出",
            ),
            # 历史
            ("历史的教训", "Will Durant", "历史", "K0/1", "四楼社科阅览室", "在馆可借"),
            (
                "人类简史：从动物到上帝",
                "Yuval Noah Harari",
                "历史",
                "K091/23",
                "四楼社科阅览室",
                "在馆可借",
            ),
            (
                "万历十五年",
                "黄仁宇",
                "中国历史",
                "K248/11",
                "四楼社科阅览室",
                "在馆可借",
            ),
            (
                "全球通史：从史前史到21世纪",
                "L. S. Stavrianos",
                "世界历史",
                "K10/5",
                "四楼社科阅览室",
                "已借出",
            ),
            # 文学与小说
            (
                "百年孤独",
                "Gabriel García Márquez",
                "小说",
                "I538.4/1",
                "五楼文学阅览室",
                "在馆可借",
            ),
            (
                "1984",
                "George Orwell",
                "科幻小说",
                "I561.4/12",
                "五楼文学阅览室",
                "在馆可借",
            ),
            ("三体", "刘慈欣", "科幻小说", "I247.5/88", "五楼文学阅览室", "已借出"),
            ("红楼梦", "曹雪芹", "古典文学", "I242.4/1", "五楼文学阅览室", "在馆可借"),
            (
                "月亮与六便士",
                "W. Somerset Maugham",
                "小说",
                "I561.4/45",
                "五楼文学阅览室",
                "在馆可借",
            ),
            ("活着", "余华", "当代文学", "I247.5/23", "五楼文学阅览室", "在馆可借"),
            # 经济与管理
            (
                "经济学原理",
                "N. Gregory Mankiw",
                "经济学",
                "F0/42",
                "四楼社科阅览室",
                "在馆可借",
            ),
            (
                "思考，快与慢",
                "Daniel Kahneman",
                "心理学/经济学",
                "F019.3/7",
                "四楼社科阅览室",
                "已借出",
            ),
            (
                "资本论",
                "Karl Marx",
                "政治经济学",
                "A811.2/1",
                "四楼社科阅览室",
                "在馆可借",
            ),
            (
                "基业长青",
                "Jim Collins",
                "管理学",
                "F270/112",
                "四楼社科阅览室",
                "在馆可借",
            ),
            # 科学与科普
            (
                "时间简史",
                "Stephen Hawking",
                "物理学",
                "P159/2",
                "二楼自然科学阅览室",
                "在馆可借",
            ),
            (
                "自私的基因",
                "Richard Dawkins",
                "生物学",
                "Q343.1/5",
                "二楼自然科学阅览室",
                "在馆可借",
            ),
            ("宇宙", "Carl Sagan", "天文学", "P1/18", "二楼自然科学阅览室", "已借出"),
            # 艺术与哲学
            (
                "艺术的故事",
                "E. H. Gombrich",
                "艺术史",
                "J09/3",
                "六楼艺术与哲学阅览室",
                "在馆可借",
            ),
            (
                "苏菲的世界",
                "Jostein Gaarder",
                "哲学",
                "B08/12",
                "六楼艺术与哲学阅览室",
                "在馆可借",
            ),
            (
                "理想国",
                "Plato",
                "哲学",
                "B502.232/1",
                "六楼艺术与哲学阅览室",
                "在馆可借",
            ),
        ]
        cursor.executemany(
            "INSERT INTO books (title, author, topic, call_number, location, status) VALUES (?, ?, ?, ?, ?, ?)",
            test_data,
        )
        conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

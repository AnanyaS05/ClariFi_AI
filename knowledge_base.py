# Step 1: Set up the Knowledge Base and Implement Search Tools
# We will begin with a small corpus of a dataset as an example, and show how we can build a search method that finds the most related document given an input query. 
# We will use this corpus as the database and the method as the action space for building a GPT-based agent.


# First, let's import necessary packages
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import json, math, re, textwrap, random, os, sys
import math
from collections import Counter, defaultdict

# A toy corpus mimicing the documents of Wikipedia
CORPUS = [
    {
        "id": "doc1",
        "title": "Nestle Financial Statements",
        "text": (
           """Nestlé is one of the largest food and beverage companies in the world, operating in 185 countries and employing around 277,000 people. 
           The brand portfolio includes many products people buy daily, such as coffee (Nescafé, Nespresso), chocolate (KitKat), bottled water, pet foods 
           like Purina, baby formula, and a wide range of health and nutrition products. Because the company is so widespread and diversified, its financial 
           statements provide a clear picture of how people are buying and consuming food around the world. In 2024, Nestlé reported total sales of about 91.4 
           billion Swiss francs. This is slightly lower than the previous year, but the company still earned a very strong profit of about 10.9 billion Swiss 
           francs, which means Nestlé remained highly profitable despite sales being a bit softer than the year before. Even though total sales decreased slightly 
           compared to 2023, Nestlé managed to maintain strong profit margins. This means that even with fewer total purchases, the company controlled its costs 
           effectively and priced its products in a way that protected earnings. Essentially, Nestlé earned about 12 cents of pure profit for every dollar of 
           product it sold, which is a very strong position in the food industry where margins are often much lower. The fact that Nestlé kept profitability high 
           suggests strong brand power — customers continue to pay for its products even when budgets are tight. When looking at where the money came from 
           geographically, the United States remained Nestlé’s most important market. North America generated over 25 billion Swiss francs in sales, while 
           Europe contributed just under 19 billion. Asia, Oceania, and Africa together produced around 16.8 billion. Latin America added nearly 12 billion, 
           with Brazil and Mexico playing particularly strong roles. Greater China contributed about 5 billion. Some regions, such as Europe and China, saw 
           slower consumer spending because of inflation and economic uncertainty, which helps explain part of the slight drop in overall sales. By product 
           category, Nestlé’s strongest-performing businesses continue to be coffee and pet food. Coffee brands like Nescafé and Nespresso, as well as Purina 
           pet food products, brought in the largest share of revenue and showed resilience even in weaker consumer environments. Nutrition and health science 
           products, which include vitamins, supplements, and specialized medical nutrition, also contributed significantly. Meanwhile, bottled water, chocolate,
           and cooking products remain stable but less dominant parts of the business. The performance of Purina in particular shows how much people continue to 
           spend on their pets, even when tightening budgets in other parts of their lives. In terms of financial structure, Nestlé has far more debt than cash, 
           with around 5.6 billion Swiss francs in cash compared to over 63 billion in debt. However, this is not necessarily a problem. Large companies often 
           use debt strategically to fund expansion, research, acquisitions, and shareholder returns. Nestlé also returned a significant amount of money to its 
           shareholders in 2024, paying out about 7.8 billion Swiss francs in dividends and spending around 4.7 billion on buying back its own shares. 
           Share buybacks reduce the number of shares in circulation, which typically increases the value of each remaining share. The company did not make any 
           major disposals in 2024, meaning it did not sell off any big divisions or brands. However, it did make strategic acquisitions, including buying Grupo 
           CRM, a premium chocolate company in Brazil, and acquiring global rights to a microbiome-based health treatment from Seres Therapeutics. 
           These purchases show Nestlé’s strategy of investing in growing niches — premium sweets in developing markets and advanced medical nutrition in 
           healthcare markets.Overall, the story of 2024 for Nestlé is one of stability, discipline, and long-term positioning rather than dramatic expansion. 
           Even though total sales did not grow, the company protected its profitability, strengthened its core product categories, and allocated substantial 
           value back to shareholders. Nestlé remains financially strong, widely diversified, and strategically positioned in both everyday consumer food and 
           the growing health and wellness sector. It continued to demonstrate the power of trusted global brands, efficient operations, and strong pricing 
           influence in the marketplace.

"""
        ),
    },

    {
        "id": "doc2",
        "title": "XYZ, Inc. Financial Statements",
        "text": (
            """This balance sheet shows the financial position of XYZ, Inc. as of December 31, 2018. A balance sheet is used to understand what a company owns, what it owes,
              and the amount that belongs to the owners. The first section lists the company’s assets, which are the resources it controls that have value. 
              The company has a total of about $6.86 million in assets. Of that, around $5.36 million are current assets, meaning assets that are expected to be used or turned into cash 
              within a year. These include almost $900,000 in cash held in checking, savings, and petty cash. The company also has about $3.59 million in accounts receivable, which 
              represents money owed to the business by customers, and includes $589,791 in work-in-process, meaning work not yet completed. In addition, the business has prepaid expenses 
              like rent and insurance, which total about $274,321. These are payments made in advance for services the company will receive later.
              The company also owns non-current assets, which are long-term items expected to provide value for more than a year. These total about $1.5 million and include things 
              like computer equipment, office furniture, field equipment, real estate, and improvements made to leased spaces. There is also about $240,031 categorized as other long-term 
              assets. Together, the current and non-current assets make up the company’s total asset base of $6,858,029. The balance sheet then lists the company’s liabilities, which 
              represent debts and obligations owed to others. The company has $2.02 million in current liabilities, which are debts due within the next year. These include accounts payable, 
              deferred taxes, money borrowed from a line of credit, and the portion of long-term debt that must be repaid within the coming year. There are also other miscellaneous 
              short-term obligations. Beyond that, the company has $870,970 in non-current liabilities, meaning longer-term debt and other obligations not due in the near term. 
              Altogether, the company owes $2,887,230 to lenders and creditors. Whatever remains after subtracting liabilities from assets represents the company’s equity, or the value 
              that belongs to the owners. XYZ, Inc. has an equity total of $3,970,799. This is made up of money that owners originally put into the business, shares the company has 
              repurchased (called treasury stock, which reduces equity), and retained earnings, which reflect profits the company has accumulated over time and kept in the business rather 
              than paying out.In simple terms, this balance sheet shows a company that has more assets than debts, meaning it is financially positive. The majority of its value comes from
              money owed to it by customers and from profits it has retained over time. The company appears to be operating with a healthy level of owner value relative to its debts, 
                which generally indicates financial stability at the time of the report.

"""
        ),
    },
    {
        "id": "doc3",
        "title": "Statement of Cash Flows",
        "text": (
            """
            This document is a Statement of Cash Flows, which is a financial report that shows how cash moves into and out of a business during a given time period. 
            While the income statement shows profit and the balance sheet shows what the company owns and owes, the cash flow statement shows something even more practical: 
            how much actual cash the business has available to spend. This is important because a company can appear profitable on paper but still run into trouble if its cash is 
            tied up in unpaid invoices or inventory. The statement begins with the Beginning Cash on Hand, which represents the amount of money the business had at the very 
            start of the reporting period. From there, it adds all forms of Cash Receipts, which are any inflows of cash. These may include cash sales, payments collected from customers 
            who previously owed money, interest earned, tax refunds, or any other cash received. Adding up all these items shows the Total Cash Receipts, which tells us how much cash came 
            into the business during the period. Next, the statement lists Cash Payments, which are all the ways cash leaves the business. It starts with Cost of Goods Sold, which includes 
            the direct costs of producing products or delivering services, such as materials, payroll for production workers, supplies, and related taxes. After that, the statement includes
            Operating Expenses, which are the everyday costs of running the business — advertising, office supplies, payroll processing, utilities, rent, bank fees, insurance, travel, dues,
            software, and many other normal operating costs. These are the expenses you must pay to keep the business functioning, even if no products are being sold at that moment.
            There is also a section for Other Expense Payments, such as loan payments (interest), distributions paid to business owners, or income tax payments. When all these outgoing cash
            payments are totaled, the business arrives at Total Cash Payments. The difference between all the incoming cash and outgoing cash is called the Net Cash Change, which shows whether 
            the business gained cash or lost cash over the period. Finally, the ending figure, Cash Position (end of month), tells us how much cash the business has left after considering all inflows
            and outflows. This ending cash balance is critical because it reflects the business’s real ability to operate — to pay employees, pay rent, handle unexpected costs, and make investment decisions. 
            In other words, this is the company's actual “breathing room.” In simple terms, the purpose of the cash flow statement is to help a business owner understand whether their 
            company is building cash, burning cash, or simply maintaining its cash level. It reveals the business’s true financial health more clearly than profit numbers alone. 
            A company can be profitable but still run out of cash; this report prevents that by showing the real movement of money.

            """
        ),
    },
    {
        "id": "doc4",
        "title": "Smart Money Stock Screen",
        "text": (
            """
This study looks at whether the stock recommendations published in the Wall Street Journal’s weekly Smart Money Stock Screen (SMSS) actually affect stock prices, trading activity, and investor behavior. The SMSS selects a small group of stocks each week and labels them as either stocks to buy (long positions) or stocks to avoid or short (short positions). The key question the researchers wanted to answer was: If everyone has access to these recommendations for free, do they still move the market?
The researchers find that yes — these recommendations do move the market. Stocks that are recommended as buys experience a positive price increase of about 1.17% after the recommendation is published. On the other hand, stocks that are recommended as shorts experience a larger price drop, about -5.85%, after the article comes out. This means negative recommendations have an even stronger effect than positive ones. Investors react noticeably to the information, even though it is publicly available at no cost.

The paper also finds that trading volume increases, especially around the time the article is being prepared and released. Interestingly, the study finds signs that the market may begin reacting to the recommendation before it actually becomes public, particularly around five days before publication. This suggests one of two things may be happening:
1.	Information leaks while the article is being researched and edited, or

2.	Other investors independently run similar stock screens and begin trading earlier, leading to movement before the official recommendation appears.


Unlike many similar studies that examine other WSJ columns or TV shows such as Mad Money, this study finds that the price reactions to SMSS do not reverse later. In many cases, after a recommendation appears, the price briefly jumps or drops and then returns to normal. But here, the movement is more permanent. This suggests the stocks being recommended may actually carry useful information, rather than just short-term hype.

The researchers connect their findings to a well-known concept in finance called “informational efficiency.” In theory, if markets were perfectly efficient, publicly available information would not help investors earn extra profits because the price would already reflect it. But this study supports the idea that markets are not perfectly efficient. Even free, publicly available recommendations — when published by a trusted, widely read source like the Wall Street Journal — can still cause real, lasting price changes.

In Simple Terms:
●	When the Wall Street Journal says “buy this stock,” the price goes up.

●	When it says “avoid or short this stock,” the price goes down — and much more strongly.

●	These price movements happen quickly, and unlike many news reactions, they don’t fade away later.

●	Trading activity increases, sometimes even before the article is released, suggesting information leaks or investor anticipation.

Why This Matters to an Investor or Trader:
This means the Smart Money Stock Screen has real predictive or influential power.
 A trader could potentially create a strategy based on these recommendations — particularly the short recommendations, which show larger, more profitable reactions.
It also shows that who says the information matters, not just what the information is.
 A well-known source can move markets.


"""
        ),
    },
    {
        "id": "doc5",
        "title": "Personal Financial Statement",
        "text": (
            """
This document is a Personal Financial Statement. It is used when a person is applying for a loan, business financing, a mortgage, a line of credit, or any situation where a bank or lender 
needs to understand an individual’s financial strength and stability. Instead of focusing on a business, this form reflects the financial life of the individual — what you own, what you 
owe, how much income you receive, and how much debt or obligation you are responsible for. It functions like a personal balance sheet, showing your net worth at a specific date.
The first part of the form collects personal information, such as your name, address, phone numbers, and whether the assets listed are held individually or jointly (for example, shared 
with a spouse). The form specifies that if assets are owned jointly, both owners must sign, because both parties have legal claim to the assets and may also share the debt responsibilities. 
This ensures accuracy and legal responsibility in the event of a loan or credit approval.
The next section lists your assets, which are everything you own that has financial value. These may include cash in checking and savings accounts, retirement accounts like IRAs, money 
other people owe you (accounts receivable), life insurance with cash surrender value, investments such as stocks and bonds, real estate you own, your vehicle(s), and other personal 
valuables. The form requires you to list each category of asset and its value, so that the total can be calculated. This total represents your total financial resources.
The form then lists your liabilities, which are your debts and financial obligations. These include credit card balances, personal loans, auto loans, mortgages on real estate you own, 
unpaid taxes, and any other debts. The form may also ask you to describe these debts in more detail in later sections. After listing these, the lender subtracts your total liabilities 
from your total assets to determine your net worth — which is the true measure of your financial position.
In Section 1, the form asks about your sources of income, including salary, investment earnings, rental income, or any other regular incoming money. It also asks about contingent liabilities — 
potential debts you may not be paying now but could become responsible for later — such as co-signing loans, legal disputes, or tax claims. These questions matter because they show 
whether your income is reliable and whether any upcoming responsibilities could weaken your financial position.
Additional sections provide room to explain details, such as loans you owe to banks or other creditors, investments in stocks and bonds, real estate owned, unpaid taxes, personal property,
and life insurance policies. These sections break down your financial picture into specifics, allowing a lender to see what type of assets you hold, how liquid they are, and what obligations
are tied to them. For example, real estate may have high value but may also have a mortgage attached. Stocks may have market value that can fluctuate. Life insurance may have a cash value that 
could be accessed if necessary.
Finally, the form requires your signature, confirming that everything you have listed is true and accurate. By signing, you also give the lender permission to verify the information, 
such as checking credit reports, bank balances, and tax history. The document also reminds you that intentionally providing false information can have serious legal consequences.
________________________________________
In simple terms:
 This form gives a complete snapshot of your financial situation, showing:
What you own
 What you owe
 How much income you actually have
 And your true net worth
It allows a lender to decide whether you are a safe and financially responsible borrower.


"""
),
    },
]

# Then, we design a simple search method based on TF-IDF to retrieve information from the corpus.

# TF-IDF (Term Frequency–Inverse Document Frequency) is a method to find the most relevant passages for a query.

# 1. We will tokenize each document and the query into words.
# 2. We will compute TF (Term Frequency) to measure how often a word appears in a document. More frequent indicates more important within that document.
# 3. We will compute IDF (Inverse Document Frequency), which is used to downweight words that are common across many documents, like “the” or “and,” and upweight rarer words.
# 4. We will compute TF-IDF vectors (containing the TF-IDF score for each word) for both documents and the query, then compute cosine similarity between the query vector and each document vector.
# 5. We will compute cosine similarity between the query vector and each document vector.
# 6. We will implement a search method that finds the documents with the highest similarity scores as the top-k search results.
# 7. We note that this action space can mostly only retrieve a small part of a passage based on the exact passage name, which is weaker than state-of-the-art retrievers. The purpose is to simulate how the search method in Wikipedia and make models to retrieve via reasoning in language.

# As an extension of the project, you can redefine the search method in this code snippet to incorporate a more powerful search method.

# 1.  Tokenize the document into words
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

#     Get all the words of each document in the corpus
DOC_TOKENS = [tokenize(d["title"] + " " + d["text"]) for d in CORPUS]

#     Get all the words from the corpus
VOCAB = sorted(set(t for doc in DOC_TOKENS for t in doc))


# 2.  Compute term frequency (TF) for each doc
def compute_tf(tokens: List[str]) -> Dict[str, float]:
    # Input: A list of all the words in a document
    # Output: A dictionary of the frequency of each word
    
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)
    # normalized term frequency
    tf: Dict[str, float] = {t: counts[t] / total for t in counts}
    return tf


# 3.   Compute the document frequency across corpus: how many docs does a word appear?
def compute_df(doc_tokens: List[List[str]]) -> Dict[str, float]:
    # Input: A list of lists of tokens in each document
    # Output: A dictionary of the counts of each word appearing across the documents

    # ===== TODO =====
    # implement the function to compute document frequency: count of the word appearing in the documents
    
    # Count number of documents in which each token appears (unique tokens per document)
    df: Dict[str, int] = defaultdict(int)
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1

    # Ensure every term in VOCAB appears in the DF mapping (0 if not present)
    for t in VOCAB:
        if t not in df:
            df[t] = 0

    return dict(df)

#     Compute the inverse document frequency (higher for rarer terms), in which we use a smoothed variant
DF = compute_df(DOC_TOKENS) # Get the DF
N_DOC = len(DOC_TOKENS) # number of docs
IDF = {t: math.log((N_DOC + 1) / (DF[t] + 0.5)) + 1 for t in VOCAB} # Inverse document frequency



# 4.   We compute TF-IDF vectors for each document, which is the product between
def tfidf_vector(tokens: List[str]) -> Dict[str, float]:
    # Input: A list of words in a document
    # Output: A dictionary of tf-idf score of each word
    tf = compute_tf(tokens)
    vec = {t: tf[t] * IDF.get(t, 0.0) for t in tf}
    return vec

DOC_VECS = [tfidf_vector(tokens) for tokens in DOC_TOKENS]


# 5.   We compute the cosine similarity for the search
def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Inputs: Two dictrionaries of tf-idf vectors of two document
    # Output: The cosine similarity scalar between the two vector

    if not a or not b:
        return 0.0

    # Compute the cosine similarity between two tf-idf vectors
    # Notice that they are two dictionaries and could have missing keys
    
    # compute dot product
    
    # compute norms
    # dot product over intersection keys
    dot = 0.0
    for k, v in a.items():
        if k in b:
            dot += v * b[k]

    # norms
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# 6.   We implement a search method based on the cosine similarity, which finds the documents with the highest similarity scores as the top-k search results.
def search_corpus(query: str, k: int = 3) -> List[Dict[str, Any]]:
    qvec = tfidf_vector(tokenize(query))
    scored = [(cosine(qvec, v), i) for i, v in enumerate(DOC_VECS)]
    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:k]:
        d = CORPUS[idx].copy()
        d["score"] = float(score)
        results.append(d)
    return results

#       Integrate the search method as a tool
def tool_search(query: str, k: int = 3) -> Dict[str, Any]:
    hits = search_corpus(query, k=k)
    # Return a concise, citation-friendly payload
    return {
        "tool": "search",
        "query": query,
        "results": [
            {"id": h["id"], "title": h["title"], "snippet": h["text"][:240] + ("..." if len(h["text"]) > 240 else "")}
            for h in hits
        ],
    }

TOOLS = {
    "search": {
        "schema": {"query": "str", "k": "int? (default=3)"},
        "fn": tool_search
    },
    "finish": {
        "schema": {"answer": "str"},
        "fn": lambda answer: {"tool": "finish", "answer": answer}
    }
}

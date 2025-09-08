class RouterRetriever:
    def __init__(self, listing_retriever, market_retriever, logger):
        self.listing_retriever = listing_retriever
        self.market_retriever = market_retriever
        self.logger = logger
    
    def get_relevant_documents(self, query: str): #not the same as the vectorstore.as_retriever().get_relevant_documents(query) its just a wrapper class
        choice = route_query(query)
        self.logger.info(f"choosing router...")
        retriever = self.listing_retriever if choice == "listing" else self.market_retriever
        self.logger.info(f"Routing query -> {choice} retriever")

        return retriever.get_relevant_documents(query) #this will call the actual vectorstore.as_retriever().get_relevant_documents(query) and return the data


def route_query(query: str) -> str:
    listings_keywords = ["list", "listed", "company", "ticker", "symbol", "nyse", "nasdaq", "iex", "cboe"]
    market_keywords = ["market", "performance", "trend", "return", "gain", "loss", "index", "s&p"]

    q = query.lower()
    if any(word in q for word in listings_keywords):
        return "listing"
    elif any(word in q for word in market_keywords):
        return "market"
    else:
        return "market"
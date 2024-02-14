2. 
df = read.csv("C:\\Users\\alyam\\PycharmProjects\\Principal Final Deliverables\\Data\\growth-data-all.csv",stringsAsFactors = FALSE)
library(udpipe)
library(textrank)
## First step: Take the Spanish udpipe model and annotate the text. Note: this takes about 3 minutes


ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(ud_model$file_model)
x <- udpipe_annotate(ud_model, x = df$Sentences)
x <- as.data.frame(x)



## Collocation (words following one another)
stats <- keywords_collocation(x = x, 
                              term = "token", group = c("doc_id", "paragraph_id", "sentence_id"),
                              ngram_max = 4)
## Co-occurrences: How frequent do words occur in the same sentence, in this case only nouns or adjectives
stats <- cooccurrence(x = subset(x, upos %in% c("NOUN", "ADJ")), 
                      term = "lemma", group = c("doc_id", "paragraph_id", "sentence_id"))
## Co-occurrences: How frequent do words follow one another
stats <- cooccurrence(x = x$lemma, 
                      relevant = x$upos %in% c("NOUN", "ADJ"))
## Co-occurrences: How frequent do words follow one another even if we would skip 2 words in between
stats <- cooccurrence(x = x$lemma, 
                      relevant = x$upos %in% c("NOUN", "ADJ"), skipgram = 2)
head(stats)

library(igraph)
library(ggraph)
library(ggplot2)
wordnetwork <- head(stats, 30)
wordnetwork <- graph_from_data_frame(wordnetwork)
ggraph(wordnetwork, layout = "fr") +
  geom_edge_link(aes(width = cooc, edge_alpha = cooc), edge_colour = "pink") +
  geom_node_text(aes(label = name), col = "darkgreen", size = 4) +
  theme_graph(base_family = "Arial Narrow") +
  theme(legend.position = "none") +
  labs(title = "Cooccurrences within 3 words distance", subtitle = "Nouns & Adjective")

3. 
stats <- textrank_keywords(x$lemma, 
                              relevant = x$upos %in% c("NOUN", "ADJ"), 
                              ngram_max = 8, sep = " ")
stats <- subset(stats$keywords, ngram > 1 & freq >= 5)
library(wordcloud)
wordcloud(words = stats$keyword, freq = stats$freq)

6.
stats <- merge(x, x, 
               by.x = c("doc_id", "paragraph_id", "sentence_id", "head_token_id"),
               by.y = c("doc_id", "paragraph_id", "sentence_id", "token_id"),
               all.x = TRUE, all.y = FALSE, 
               suffixes = c("", "_parent"), sort = FALSE)
stats <- subset(stats, dep_rel %in% "nsubj" & upos %in% c("NOUN") & upos_parent %in% c("ADJ"))
stats$term <- paste(stats$lemma_parent, stats$lemma, sep = " ")
stats <- txt_freq(stats$term)
library(wordcloud)
wordcloud(words = stats$key, freq = stats$freq, min.freq = 3, max.words = 100,
          random.order = FALSE, colors = brewer.pal(6, "Dark2"))

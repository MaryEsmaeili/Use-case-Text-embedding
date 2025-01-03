import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TopicVisualization:
    @staticmethod
    def display_topics(model, feature_names, num_top_words, topic_names=None):
        """
        Display topics and their top words.
        """
        for ix, topic in enumerate(model.components_):
            if not topic_names or not topic_names[ix]:
                print(f"\nTopic {ix}")
            else:
                print(f"\nTopic: '{topic_names[ix]}'")
            print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

    @staticmethod
    def plot_topic_bars(model, feature_names, num_top_words):
        """
        Plot enhanced bar charts for the top words in each topic.
        """
        topic_term_matrix = model.components_
        sns.set(style="whitegrid")
        palette = sns.color_palette("muted", len(topic_term_matrix))

        for topic_idx, topic in enumerate(topic_term_matrix):
            plt.figure(figsize=(5, 3))
            top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = topic[top_words_idx]

            # Create horizontal bar plot
            sns.barplot(x=top_weights, y=top_words, palette=[palette[topic_idx]] * num_top_words)

            # Add title and labels
            plt.title(f"Topic {topic_idx + 1} - Top Words", fontsize=16, fontweight="bold", pad=10)
            plt.xlabel("Importance", fontsize=12)
            plt.ylabel("Words", fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            # Show plot
            plt.show()
    @staticmethod
    def plot_top_words(model, feature_names, n_top_words, title):
        """
        Plot top words for each topic as bar charts.
        """
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[-n_top_words:]
            top_features = feature_names[top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7, color="skyblue")
            ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 20})
            ax.tick_params(axis="both", which="major", labelsize=15)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)

        fig.suptitle(title, fontsize=25)
        plt.subplots_adjust(top=0.9, bottom=0.05, wspace=0.9, hspace=0.3)
        plt.show()

    @staticmethod
    def process_and_plot_model(model, tfidf, feature_names, n_top_words, title):
        """
        Fit a topic modeling model and plot the top words.
        """
        model.fit(tfidf)
        TopicVisualization.plot_top_words(model, feature_names, n_top_words, title)
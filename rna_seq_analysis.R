# RNA-seq Differential Gene Expression Analysis using DESeq2

library(DESeq2)
library(ggplot2)
library(pheatmap)

# Load count matrix and metadata
counts <- read.csv("counts_matrix.csv", row.names=1)
col_data <- read.csv("metadata.csv", row.names=1)

# Create DESeq2 dataset
dds <- DESeqDataSetFromMatrix(countData=counts,
                              colData=col_data,
                              design=~condition)

# Pre-filtering
dds <- dds[rowSums(counts(dds)) > 10,]

# Run DESeq pipeline
dds <- DESeq(dds)
res <- results(dds)

# Order results by adjusted p-value
res_ordered <- res[order(res$padj),]

# Save significant results
sig_genes <- subset(res_ordered, padj < 0.05)
write.csv(as.data.frame(sig_genes), "significant_genes.csv")

# Volcano plot
res$significant <- ifelse(res$padj < 0.05 & abs(res$log2FoldChange) > 1, "yes", "no")
ggplot(as.data.frame(res), aes(x=log2FoldChange, y=-log10(pvalue), color=significant)) +
    geom_point(alpha=0.5) +
    theme_minimal() +
    ggtitle("Volcano Plot")

# Heatmap
vsd <- vst(dds, blind=FALSE)
select_genes <- rownames(sig_genes)[1:50]
pheatmap(assay(vsd)[select_genes,], cluster_rows=TRUE, show_rownames=FALSE,
         cluster_cols=TRUE, annotation_col=col_data)

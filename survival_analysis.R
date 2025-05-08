# Survival Analysis in R using clinical trial data

library(survival)
library(survminer)

# Load data
data <- read.csv("survival_data.csv")

# Create survival object
surv_obj <- Surv(time=data$time, event=data$status)

# Fit Kaplan-Meier survival curve
fit <- survfit(surv_obj ~ data$treatment)

# Plot survival curve
ggsurvplot(fit, data=data, pval=TRUE, risk.table=TRUE)

# Log-rank test
log_rank <- survdiff(surv_obj ~ data$treatment)
print(log_rank)

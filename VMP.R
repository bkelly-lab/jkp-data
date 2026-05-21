# --- 1. Load All Necessary Packages ---

# List all the packages we'll need for data wrangling, plotting, and analysis
all_packages <- c(
  "dplyr", "tidyr", "lubridate", "stringr",
  "rstudioapi",
  "frenchdata",
  "ggplot2", "scales", "corrplot", "RColorBrewer",
  "tidyquant", "readr",
  "broom",
  "patchwork",
  "lmtest",
  "sandwich",
  "gt",
  "PerformanceAnalytics",
  "xts",
  "zoo",
  "readxl"
)

# Set the CRAN mirror
options(repos = "https://cloud.r-project.org")

# Check if each package is installed, and if not, install it.
installed <- rownames(installed.packages())
for(pkg in all_packages) {
  if(! pkg %in% installed) install.packages(pkg)
}

# Load all packages
invisible(lapply(all_packages, library, character.only = TRUE))

# --- 2. Set Working Directory ---
current_path <- getActiveDocumentContext()$path
setwd(dirname(current_path))
print(getwd())

############################ PART 1.  ##########################################

download_global_q <- function(frequency = "monthly") {
  # 1. Define URLs (Updated based on global-q.org structure)
  # Note: The authors update these filenames annually. 
  # If this breaks, visit http://global-q.org/factors.html and copy the new link.
  if (frequency == "monthly") {
    url <- "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_monthly_2024.csv"
  } else if (frequency == "daily") {
    url <- "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_daily_2024.csv"
  } else {
    stop("Frequency must be 'monthly' or 'daily'")
  }
  # 2. Read Data directly from URL
  message(paste("Downloading", frequency, "data from Global-q.org..."))
  data <- read_csv(url, col_types = cols()) |>
    rename_with(tolower)
  # 3. Clean Dates and Format
  if (frequency == "monthly") {
    data <- data |>
      rename(date = year, month_num = month) |> # Global-q uses 'year' and 'month' cols
      mutate(
        date = make_date(date, month_num, 1),
        date = ceiling_date(date, "month") - days(1) # End of month
      ) |>
      select(-month_num) 
  } else {
    # Daily data usually comes in a different format, handle strictly if needed
    data <- data |>
      mutate(date = ymd(date)) # Assuming YYYYMMDD format for daily
  }
  data <- data |>
    select(date, everything())
  return(data)
}

download_aqr_bab <- function(frequency = "monthly") {
  
  # 1. Define URLs
  base_url <- "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors"
  
  if (frequency == "monthly") {
    url <- paste0(base_url, "-Monthly.xlsx")
  } else if (frequency == "daily") {
    url <- paste0(base_url, "-Daily.xlsx")
  } else {
    stop("Frequency must be 'monthly' or 'daily'")
  }
  
  # 2. Download Data
  message(paste("Downloading", frequency, "BAB data from AQR..."))
  temp_file <- tempfile(fileext = ".xlsx")
  
  tryCatch({
    # mode = "wb" is crucial for Windows/Excel files
    download.file(url, temp_file, mode = "wb", quiet = TRUE)
  }, error = function(e) {
    stop("Download failed. AQR link may have changed.")
  })
  
  # 3. Intelligent Header Detection
  # FIX: Use .name_repair = "minimal" and suppressMessages to avoid the error and console spam
  raw_preview <- suppressMessages(
    read_excel(temp_file, sheet = 1, col_names = FALSE, n_max = 50, .name_repair = "minimal")
  )
  
  # Find the row index where the first column is "Date" (Case Insensitive)
  # We convert the first column to character to avoid issues if it was read as logical/numeric
  header_row <- which(str_detect(toupper(as.character(raw_preview[[1]])), "^DATE"))[1]
  
  if (is.na(header_row)) {
    unlink(temp_file)
    stop("Could not find 'DATE' header in the AQR file. Check the file structure.")
  }
  
  # 4. Read Data starting from the detected header
  # Use .name_repair = "unique" here to ensure we have valid column names for the actual data
  data <- suppressMessages(
    read_excel(temp_file, sheet = 1, skip = header_row - 1, .name_repair = "unique")
  ) |>
    rename_with(tolower) |>
    rename(date = 1) |> # Force 1st column to be 'date'
    filter(!is.na(date)) # Remove copyright footer rows
  
  # 5. Robust Date Parsing
  data <- data |>
    mutate(
      date = case_when(
        # If Excel read it as a proper date/time object
        is.POSIXct(date) ~ as.Date(date),
        # If Excel read it as text ("01/31/1926"), parse it
        is.character(date) ~ as.Date(parse_date_time(date, orders = c("mdy", "dmy", "ymd"))),
        # Fallback
        TRUE ~ as.Date(date)
      )
    )

  # Adjust Monthly dates to End-of-Month
  if (frequency == "monthly") {
    data <- data |>
      mutate(date = ceiling_date(date, "month") - days(1))
  }
  
  unlink(temp_file)
  return(data)
}

# --- 3. Load and Prepare Fama-French & Q-Factor Data ---
start_date <- ymd("1926-01-01") 
end_date <- ymd("2015-04-30") # Likely what Moreira & Muir used 

# Download the daily Fama-French  data
factors_ff3_daily_raw <- download_french_data("Fama/French 3 Factors [Daily]")
factors_ff5_daily_raw <- download_french_data("Fama/French 5 Factors (2x3) [Daily]")
mom_data_daily_raw <- download_french_data("Momentum Factor (Mom) [Daily]")

# Download the daily q-Factor data
q_factors_daily <- download_global_q("daily")
head(q_factors_daily)

# Download BAB data from AQR
bab_data_daily <- download_aqr_bab("daily")
head(bab_data_daily)

# Extract the specific dataframes from the list objects and convert to dates
ff3_clean <- factors_ff3_daily_raw$subsets$data[[1]] |>
  mutate(date = ymd(date))

ff5_clean <- factors_ff5_daily_raw$subsets$data[[1]] |>
  select(date, RMW, CMA) |>
  mutate(date = ymd(date))

mom_clean <- mom_data_daily_raw$subsets$data[[1]] |>
  mutate(date = ymd(date)) |>
  filter(date > ymd("1926-11-30")) # Drop the 1st month, as it is not a full month

q_factors_clean <- q_factors_daily |>
  filter(date > ymd("1967-04-30")) # Drop data before April 1967 to likely match MM2017

bab_clean <- bab_data_daily |>
  mutate(date = ymd(date)) |>
  mutate(across(-date, ~ . * 100))

# Merge and clean
factors_daily <- ff3_clean |>
  # Join only the new factors (RMW, CMA) from FF5 to avoid duplicating RF/Mkt/SMB/HML
  left_join(ff5_clean, by = "date") |>
  # Join Momentum
  left_join(mom_clean, by = "date") |>
  # Join q-Factors
  left_join(q_factors_clean |> select(date, ROE = r_roe, IA = r_ia), by = "date") |>
  # Join BAB
  left_join(bab_clean |> select(date, BAB = usa), by = "date") |>
  mutate(
    date = ymd(date),
    # Convert all factors to numeric and scale
    across(c(RF, `Mkt-RF`, SMB, HML, RMW, CMA, Mom, ROE, IA, BAB), ~as.numeric(.) / 100)
  ) |>
  # Explicitly select the final column order
  select(date, RF, `Mkt-RF`, SMB, HML, RMW, CMA, Mom, ROE, IA, BAB) |>
  rename_with(str_to_lower) |>
  rename(mkt_excess = `mkt-rf`)


# Filter the date range
factors_daily <- factors_daily |>
  filter(date >= start_date & date <= end_date)


### STEPS 1 & 2a: Calculate Monthly Returns and Variance ###
# 1. Calculate monthly variance from the daily data.
# 2. Compound the daily returns to get the monthly return.
data_monthly <- factors_daily |>
  mutate(
    year_month = floor_date(date, "month"), # Create a helper column for grouping
  ) |>
  group_by(year_month) |>
  summarize(
    date = max(date), # Use end-of-month date as the identifier for this summary
    
    # --- From Step 1 ---
    mkt_excess_var = var(mkt_excess, na.rm = TRUE),
    smb_var = var(smb, na.rm = TRUE),
    hml_var = var(hml, na.rm = TRUE),
    rmw_var = var(rmw, na.rm = TRUE),
    cma_var = var(cma, na.rm = TRUE),
    mom_var = var(mom, na.rm = TRUE),
    roe_var = var(roe, na.rm = TRUE),
    ia_var = var(ia, na.rm = TRUE),
    bab_var = var(bab, na.rm = TRUE),
    

    n_days = n(), # Count the number of trading days
    
    # --- From Step 2a ---
    # Compound the daily total returns to get the monthly total return
    rf_comp = prod(1 + rf) - 1,
    mkt_return_comp = prod(1 + mkt_excess + rf) - 1 - rf_comp,
    smb_return_comp = prod(1 + smb) - 1,
    hml_return_comp = prod(1 + hml) - 1,
    rmw_return_comp = prod(1 + rmw) - 1,
    cma_return_comp = prod(1 + cma) - 1,
    mom_return_comp = prod(1 + mom) - 1,
    roe_return_comp = prod(1 + roe) - 1,
    ia_return_comp = prod(1 + ia) - 1,
    bab_return_comp = prod(1 + bab) - 1,

    .groups = "drop"
  ) |>
  select(-year_month)




### STEP 2 (cont.): Construct the Volatility-Managed Portfolio ###

# Lag the variance.
data_unscaled <- data_monthly |>
  mutate(
    mkt_excess_var_lag = lag(mkt_excess_var),
    smb_var_lag = lag(smb_var),
    hml_var_lag = lag(hml_var),
    rmw_var_lag = lag(rmw_var),
    cma_var_lag = lag(cma_var),
    mom_var_lag = lag(mom_var),
    roe_var_lag = lag(roe_var),
    ia_var_lag = lag(ia_var),
    bab_var_lag = lag(bab_var)

  ) |>
  filter(!is.na(mkt_excess_var_lag)) |> # Drop the first month
  mutate(
    mkt_excess_unscaled = mkt_return_comp / mkt_excess_var_lag,
    smb_unscaled = smb_return_comp / smb_var_lag,
    hml_unscaled = hml_return_comp / hml_var_lag,
    rmw_unscaled = rmw_return_comp / rmw_var_lag,
    cma_unscaled = cma_return_comp / cma_var_lag,
    mom_unscaled = mom_return_comp / mom_var_lag,
    roe_unscaled = roe_return_comp / roe_var_lag,
    ia_unscaled = ia_return_comp / ia_var_lag,
    bab_unscaled = bab_return_comp / bab_var_lag
  )

# Calculate the scaling constant 'c'
# c = vol(original portfolio) / vol(unscaled portfolio)
c_mkt_excess <- sd(data_unscaled$mkt_return_comp, na.rm = TRUE) / sd(data_unscaled$mkt_excess_unscaled, na.rm = TRUE)
c_smb <- sd(data_unscaled$smb_return_comp, na.rm = TRUE) / sd(data_unscaled$smb_unscaled, na.rm = TRUE)
c_hml <- sd(data_unscaled$hml_return_comp, na.rm = TRUE) / sd(data_unscaled$hml_unscaled, na.rm = TRUE)
c_rmw <- sd(data_unscaled$rmw_return_comp, na.rm = TRUE) / sd(data_unscaled$rmw_unscaled, na.rm = TRUE)
c_cma <- sd(data_unscaled$cma_return_comp, na.rm = TRUE) / sd(data_unscaled$cma_unscaled, na.rm = TRUE)
c_mom <- sd(data_unscaled$mom_return_comp, na.rm = TRUE) / sd(data_unscaled$mom_unscaled, na.rm = TRUE)
c_roe <- sd(data_unscaled$roe_return_comp, na.rm = TRUE) / sd(data_unscaled$roe_unscaled, na.rm = TRUE)
c_ia <- sd(data_unscaled$ia_return_comp, na.rm = TRUE) / sd(data_unscaled$ia_unscaled, na.rm = TRUE)
c_bab <- sd(data_unscaled$bab_return_comp, na.rm = TRUE) / sd(data_unscaled$bab_unscaled, na.rm = TRUE)

# Apply the scaling to get the final managed portfolio returns
factors_vol_managed <- data_unscaled |>
  mutate(
    mkt_excess_managed = c_mkt_excess * mkt_excess_unscaled,
    smb_managed = c_smb * smb_unscaled,
    hml_managed = c_hml * hml_unscaled,
    rmw_managed = c_rmw * rmw_unscaled,
    cma_managed = c_cma * cma_unscaled,
    mom_managed = c_mom * mom_unscaled,
    roe_managed = c_roe * roe_unscaled,
    ia_managed = c_ia * ia_unscaled,
    bab_managed = c_bab * bab_unscaled
  )

print(paste("Scaling constant (Market exc.) 'c':", round(c_mkt_excess, 6)))
print(paste("Original Portfolio SD (Market exc.):", round(sd(factors_vol_managed$mkt_return_comp), 6)))
print(paste("Managed Portfolio SD (Market exc.): ", round(sd(factors_vol_managed$mkt_excess_managed), 6)))

print(paste("Original Portfolio SD (Mom exc.):", round(sd(factors_vol_managed$mom_return_comp, na.rm=TRUE), 6)))
print(paste("Managed Portfolio SD (Mom exc.): ", round(sd(factors_vol_managed$mom_managed, na.rm=TRUE), 6)))




### STEP 3: Sort Months into Variance Quintiles ###
# Group all months into 5 buckets (quintiles) based on their lagged variance
factors_vol_managed <- factors_vol_managed |>
  mutate(
    var_quintile = ntile(mkt_excess_var_lag, 5),
    # Create nice labels for the plots
    var_quintile_labeled = factor(
      var_quintile,
      levels = 1:5,
      labels = c("Low Vol", "2", "3", "4", "High Vol"),
      ordered = TRUE
    )
  )

print("Count of months in each variance quintile:")
print(table(factors_vol_managed$var_quintile_labeled))


### STEP 4: Reproduce Figures and Tables ###

# --- 4.A: Load NBER Recession Data ---
# Load theme returns from RData + NBER data
load("nber_data_raw.RData")

nber_data <- nber_data_raw |>
  mutate(
    date = ymd(observation_date), 
    date_join = floor_date(date, "month"), # Create a key for joining (1st of month)
    us_recession = as.numeric(USREC) 
  ) |>
  # Critical Step: Deduplicate to ensure only one row per month exists
  group_by(date_join) |>
  summarize(us_recession = max(us_recession, na.rm = TRUE), .groups = "drop")

# Add the recession indicator to our main data frame
factors_vol_managed <- factors_vol_managed %>%
  mutate(
    date_join = floor_date(date, "month") # Create the matching key
  ) %>%
  left_join(nber_data, by = "date_join") %>%
  select(-date_join) 


# --- 4.B: Summary Statistics for Plots ---
# Calculate all the stats needed for the Figure 1 plots
quintile_summary <- factors_vol_managed |>
  group_by(var_quintile_labeled) |>
  summarize(
    mean_ret_orig_ann = mean(mkt_return_comp) * 12,
    mean_ret_man_ann = mean(mkt_excess_managed) * 12,
    sd_ret_orig_ann = sd(mkt_return_comp) * sqrt(12),
    sd_ret_man_ann = sd(mkt_excess_managed) * sqrt(12),
    mean_var_orig = mean(mkt_return_comp) / var(mkt_return_comp),
    mean_var_man = mean(mkt_excess_managed) / var(mkt_excess_managed),
    prob_recession = mean(us_recession, na.rm = TRUE),
    .groups = "drop"
  )

print("--- Summary Statistics for Figure 1 ---")
print(quintile_summary)


# --- 4.C: Create Bar Charts (Replication of Figure 1) ---
# This block plots the original portfolio stats to replicate Figure 1.

# Plot 1: Average Return (Original Portfolio)
p1_orig <- ggplot(quintile_summary, aes(x = var_quintile_labeled, y = mean_ret_orig_ann * 100)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  # Set Y-axis scale to match the paper (0 to 12)
  scale_y_continuous(limits = c(0, 12), breaks = seq(0, 12, 2)) +
  labs(
    title = "Average Return",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.x = element_blank())

# Plot 2: Standard Deviation (Original Portfolio)
p2_orig <- ggplot(quintile_summary, aes(x = var_quintile_labeled, y = sd_ret_orig_ann * 100)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  # Set Y-axis scale to match the paper (0 to 40)
  scale_y_continuous(limits = c(0, 40), breaks = seq(0, 40, 10)) +
  labs(
    title = "Standard Deviation",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.x = element_blank())

# Plot 3: Ratio E[R]/Var(R) (Original Portfolio)
p3_orig <- ggplot(quintile_summary, aes(x = var_quintile_labeled, y = mean_var_orig)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  # Set Y-axis scale to match the paper (0 to 8)
  scale_y_continuous(limits = c(0, 8), breaks = seq(0, 8, 2)) +
  labs(
    title = "E[R]/Var(R)",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.x = element_blank())

# Plot 4: Probability of Recession
p4_orig <- ggplot(quintile_summary, aes(x = var_quintile_labeled, y = prob_recession)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  # Set Y-axis scale to match the paper (0 to 0.5)
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), 
                     limits = c(0, 0.5), breaks = seq(0, 0.5, 0.1)) +
  labs(
    title = "Probability of Recession",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.x = element_blank())

# --- Combine all 4 plots using patchwork ---
figure_1_replication <- (p1_orig + p2_orig) / (p3_orig + p4_orig)


# Print the replicated Figure 1
print(figure_1_replication)


# --- 4.C: Create Bar Charts (Comparison) ---
# This is the comparison plot block (Original vs. Managed)

# Define the custom colors
my_colors <- c("Managed" = "#80bef1", "Original" = "darkblue")

# Plot 1: Average Monthly Return (Annualized)
p1_data <- quintile_summary |>
  select(var_quintile_labeled, mean_ret_orig_ann, mean_ret_man_ann) |>
  pivot_longer(
    cols = -var_quintile_labeled,
    names_to = "portfolio",
    values_to = "return",
    names_prefix = "mean_ret_"
  ) |>
  mutate(portfolio = if_else(portfolio == "orig_ann", "Original", "Managed"))

p1 <- ggplot(p1_data, aes(x = var_quintile_labeled, y = return, fill = portfolio)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = my_colors) +
  labs(
    title = "Average Return",
    x = "Variance Quintile",
    y = "Annualized Return",
    fill = "Portfolio"
  ) +
  theme_minimal() +
  theme(panel.grid.major.x = element_blank())

# Plot 2: Standard Deviation of Returns (Annualized)
p2_data <- quintile_summary |>
  select(var_quintile_labeled, sd_ret_orig_ann, sd_ret_man_ann) |>
  pivot_longer(
    cols = -var_quintile_labeled,
    names_to = "portfolio",
    values_to = "std_dev",
    names_prefix = "sd_ret_"
  ) |>
  mutate(portfolio = if_else(portfolio == "orig_ann", "Original", "Managed"))

p2 <- ggplot(p2_data, aes(x = var_quintile_labeled, y = std_dev, fill = portfolio)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = my_colors) + 
  labs(
    title = "Standard Deviation",
    x = "Variance Quintile",
    y = "Annualized Std. Dev.",
    fill = "Portfolio"
  ) +
  theme_minimal() +
  theme(panel.grid.major.x = element_blank())

# Plot 3: Ratio E[R]/Var(R)
p3_data <- quintile_summary |>
  select(var_quintile_labeled, mean_var_orig, mean_var_man) |>
  pivot_longer(
    cols = -var_quintile_labeled,
    names_to = "portfolio",
    values_to = "ratio",
    names_prefix = "mean_var_"
  ) |>
  mutate(portfolio = if_else(portfolio == "orig", "Original", "Managed"))

p3 <- ggplot(p3_data, aes(x = var_quintile_labeled, y = ratio, fill = portfolio)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
  scale_fill_manual(values = my_colors) +
  labs(
    title = "E[R] / Var(R)",
    x = "Variance Quintile",
    y = "Ratio (Monthly)",
    fill = "Portfolio"
  ) +
  theme_minimal() +
  theme(panel.grid.major.x = element_blank())

# Plot 4: Probability of being in a Recession
p4 <- ggplot(quintile_summary, aes(x = var_quintile_labeled, y = prob_recession)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Probability of Recession",
    x = "Variance Quintile",
    y = "Probability"
  ) +
  theme_minimal() +
  theme(panel.grid.major.x = element_blank())

# --- Combine all 4 plots using patchwork ---
combined_plot <- (p1 + p2) / (p3 + p4) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom") 

# Print the combined plot
print(combined_plot)



colnames(factors_vol_managed)

# Calculate the number of non-NA rows in factors_vol_managed$mkt_excess_managed
sum(!is.na(factors_vol_managed$bab_managed))



# --- 5. Construct Table 1 (Panel A) with Diagonal Structure ---

# 1. Define the mapping of factors (Label = c(Managed_Col, Original_Col))
#    We use the exact labels from the paper for the rows (MktRF, SMB, etc.)
factor_map <- list(
  "MktRF" = c("mkt_excess_managed", "mkt_return_comp"),
  "SMB"   = c("smb_managed", "smb_return_comp"),
  "HML"   = c("hml_managed", "hml_return_comp"),
  "Mom"   = c("mom_managed", "mom_return_comp"),
  "RMW"   = c("rmw_managed", "rmw_return_comp"),
  "CMA"   = c("cma_managed", "cma_return_comp"),
  "ROE"   = c("roe_managed", "roe_return_comp"),
  "IA"    = c("ia_managed", "ia_return_comp"),
  "BAB"   = c("bab_managed", "bab_return_comp")
)

# 2. Function to extract specific formatted cell values
get_column_data <- function(df, managed_col, orig_col, target_factor_name) {
  
  fmla <- as.formula(paste(managed_col, "~", orig_col))
  model <- lm(fmla, data = df)
  
  fmla_ff3 <- as.formula(paste(managed_col, "~ mkt_return_comp + smb_return_comp + hml_return_comp"))
  model_ff3 <- lm(fmla_ff3, data = df)

  # Robust SEs
  coeffs <- coeftest(model, vcov = vcovHAC(model))
  coeffs_ff3 <- coeftest(model_ff3, vcov = vcovHAC(model_ff3))
  stats  <- broom::glance(model)
  stats_ff3  <- broom::glance(model_ff3)
  
  # --- Format Coefficients (Beta) ---
  # The slope is the 2nd coefficient. 
  beta_val <- coeffs[2, "Estimate"]
  beta_se  <- coeffs[2, "Std. Error"]
  
  # Create the string "0.61<br>(0.05)" for the diagonal
  beta_str <- paste0(
    sprintf("%.2f", beta_val), "<br>(", sprintf("%.2f", beta_se), ")"
  )
  
  # --- Format Alpha ---
  # Alpha is the 1st coefficient. Multiply by 1200 for annual percentage.
  alpha_val <- coeffs[1, "Estimate"] * 1200
  alpha_se  <- coeffs[1, "Std. Error"] * 1200
  alpha_str <- paste0(
    sprintf("%.2f", alpha_val), "<br>(", sprintf("%.2f", alpha_se), ")"
  )

  alpha_val_ff3 <- coeffs_ff3[1, "Estimate"] * 1200
  alpha_se_ff3  <- coeffs_ff3[1, "Std. Error"] * 1200
  alpha_str_ff3 <- paste0(
    sprintf("%.2f", alpha_val_ff3), "<br>(", sprintf("%.2f", alpha_se_ff3), ")"
  )
  
  # --- Format Summary Stats ---
  n_str    <- sprintf("%d", stats$nobs)
  r2_str   <- sprintf("%.2f", stats$r.squared)
  rmse_str <- sprintf("%.2f", stats$sigma * 1200) # Annualized RMSE
  
  # --- Return a named list representing this COLUMN in the table ---
  # We initialize all factor rows to empty strings ""
  col_data <- list(
    "MktRF" = "", "SMB" = "", "HML" = "", 
    "Mom" = "", "RMW" = "", "CMA" = "",  
    "ROE" = "", "IA" = "", "BAB" = "",
    "Alpha" = alpha_str,
    "N" = n_str,
    "R2" = r2_str,
    "RMSE" = rmse_str,
    "Alpha_FF3" = alpha_str_ff3
  )
  
  # Fill ONLY the row corresponding to the current factor (The Diagonal)
  col_data[[target_factor_name]] <- beta_str
  
  return(col_data)
}

# 3. Build the data frame column by column
results_list <- list()

# Labels for the columns in the final table
col_headers <- c("Mkt", "SMB", "HML", "Mom", "RMW", "CMA", "ROE", "IA", "BAB")
names(col_headers) <- names(factor_map)

for (f_name in names(factor_map)) {
  cols <- factor_map[[f_name]]
  # Generate data for this factor's column
  col_res <- get_column_data(factors_vol_managed, cols[1], cols[2], f_name)
  results_list[[f_name]] <- col_res
}

# --- 3. Build Data Frame, Inject FX, and Define Groups ---

# Convert list to Data Frame
table_data <- bind_rows(results_list, .id = "column_factor") |>
  pivot_longer(cols = -column_factor) |>
  pivot_wider(names_from = column_factor, values_from = value) |>
  
  # 1. Create empty FX Column and Row
  mutate(FX = NA_character_) |>       
  add_row(name = "FX") |>             
  relocate(FX, .after = CMA) |>       
  
  # 2. Manually put "NA" on Diagonal AND Bottom Rows
  mutate(
    FX = case_when(
      name == "FX" ~ "NA",                              
      name %in% c("Alpha", "N", "R2", "RMSE", "Alpha_FF3") ~ "NA",   
      TRUE ~ FX                                         
    )
  ) |>

  # 3. Reorder Rows
  mutate(name = factor(name, levels = c(
    "MktRF", "SMB", "HML", "Mom", "RMW", "CMA", "FX", "ROE", "IA", "BAB",
    "Alpha", "N", "R2", "RMSE", "Alpha_FF3"
  ))) |>
  
  # 4. --- NEW: Define Panel Groups ---
  mutate(
    panel_group = ifelse(name == "Alpha_FF3", 
                         "Panel B: Alphas controlling for FF3", 
                         "Panel A: Univariate Regressions")
  ) |>
  # Ensure Panel A comes before Panel B
  mutate(panel_group = factor(panel_group, levels = c("Panel A: Univariate Regressions", "Panel B: Alphas controlling for FF3"))) |>
  
  # Sort by Group then by the specific row order we defined
  arrange(panel_group, name)

# --- 4. Create the formatted GT Table ---

final_table_replica <- table_data |>
  # --- Define the Grouping Column here ---
  gt(groupname_col = "panel_group") |>
  
  # --- Add Main Table Header ---
  tab_header(
    title = "Replication of Table I in Moreira & Muir (2017)"
  ) |>
  
  # --- Rename Columns ---
  cols_label(
    name = "", 
    MktRF = html("(1)<br>Mkt<sup>&sigma;</sup>"),
    SMB   = html("(2)<br>SMB<sup>&sigma;</sup>"),
    HML   = html("(3)<br>HML<sup>&sigma;</sup>"),
    Mom   = html("(4)<br>Mom<sup>&sigma;</sup>"),
    RMW   = html("(5)<br>RMW<sup>&sigma;</sup>"),
    CMA   = html("(6)<br>CMA<sup>&sigma;</sup>"),
    FX    = html("(7)<br>FX<sup>&sigma;</sup>"),
    ROE   = html("(8)<br>ROE<sup>&sigma;</sup>"),
    IA    = html("(9)<br>IA<sup>&sigma;</sup>"),
    BAB   = html("(10)<br>BAB<sup>&sigma;</sup>")
  ) |>
  
  # --- Rename Rows ---
  text_transform(
    locations = cells_body(columns = name),
    fn = function(x) {
      dplyr::case_when(
        x == "Alpha" ~ "Alpha (&alpha;)",
        x == "R2"    ~ "R<sup>2</sup>",
        TRUE         ~ x 
      )
    }
  ) |>
  
  # --- Interpret markdown ---
  fmt_markdown(columns = everything()) |>
  
  # --- Styling ---
  cols_align(align = "center", columns = -name) |>
  cols_align(align = "left", columns = name) |>
  
  # --- Hide off-diagonal NAs ---
  sub_missing(columns = everything(), missing_text = "") |> 
  
  tab_options(
    table.border.top.color = "black",
    table.border.top.width = px(2),
    table.border.bottom.color = "black",
    table.border.bottom.width = px(2),
    heading.align = "center",
    column_labels.border.bottom.color = "black",
    column_labels.border.bottom.width = px(1),
    table.font.size = 12,
    
    # --- Style the Panel Headers ---
    row_group.font.weight = "bold",
    row_group.background.color = "#f0f0f0" # Light gray background for Panel headers
  )

# --- Print ---
print(final_table_replica)


##### ENDED HERE #####



# --- 4.E: --- SCATTER PLOT FOR PART 1 ---
cat("\nGenerating scatter plot for Part 1...\n")

scatter_p1 <- ggplot(factors_vol_managed, aes(x = mkt_return_comp, y = mkt_excess_managed)) +
  geom_point(alpha = 0.2, color = "blue") +
  # Add the 45-degree (y=x) line for reference
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
    title = "Part 1: Managed Market vs. Original Market Returns (1926-2015)",
    x = "Original Market Excess Return",
    y = "Managed Market Return"
  ) +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

print(scatter_p1)


############################ PART 1b: Extended Sample (1926–2025) ##########################################
# Tests whether Moreira & Muir (2017) results hold post-publication.
# Reuses data already downloaded in Part 1 — no re-download needed.
# Note: q5 series (ROE, IA) end at end of 2024; FF and BAB extend through 2025.

end_date_full <- ymd("2025-12-31")

# Rebuild factors_daily for the full sample (identical transformations to Part 1)
factors_daily_full <- ff3_clean |>
  left_join(ff5_clean, by = "date") |>
  left_join(mom_clean, by = "date") |>
  left_join(q_factors_clean |> select(date, ROE = r_roe, IA = r_ia), by = "date") |>
  left_join(bab_clean |> select(date, BAB = usa), by = "date") |>
  mutate(
    date = ymd(date),
    across(c(RF, `Mkt-RF`, SMB, HML, RMW, CMA, Mom, ROE, IA, BAB), ~as.numeric(.) / 100)
  ) |>
  select(date, RF, `Mkt-RF`, SMB, HML, RMW, CMA, Mom, ROE, IA, BAB) |>
  rename_with(str_to_lower) |>
  rename(mkt_excess = `mkt-rf`) |>
  filter(date >= start_date & date <= end_date_full)

cat("Part 1b — full-sample data range:",
    format(min(factors_daily_full$date)), "to",
    format(max(factors_daily_full$date)), "\n")

# Monthly returns and variances
data_monthly_full <- factors_daily_full |>
  mutate(year_month = floor_date(date, "month")) |>
  group_by(year_month) |>
  summarize(
    date            = max(date),
    mkt_excess_var  = var(mkt_excess, na.rm = TRUE),
    smb_var         = var(smb,        na.rm = TRUE),
    hml_var         = var(hml,        na.rm = TRUE),
    rmw_var         = var(rmw,        na.rm = TRUE),
    cma_var         = var(cma,        na.rm = TRUE),
    mom_var         = var(mom,        na.rm = TRUE),
    roe_var         = var(roe,        na.rm = TRUE),
    ia_var          = var(ia,         na.rm = TRUE),
    bab_var         = var(bab,        na.rm = TRUE),
    n_days          = n(),
    rf_comp             = prod(1 + rf) - 1,
    mkt_return_comp     = prod(1 + mkt_excess + rf) - 1 - (prod(1 + rf) - 1),
    smb_return_comp     = prod(1 + smb) - 1,
    hml_return_comp     = prod(1 + hml) - 1,
    rmw_return_comp     = prod(1 + rmw) - 1,
    cma_return_comp     = prod(1 + cma) - 1,
    mom_return_comp     = prod(1 + mom) - 1,
    roe_return_comp     = prod(1 + roe) - 1,
    ia_return_comp      = prod(1 + ia)  - 1,
    bab_return_comp     = prod(1 + bab) - 1,
    .groups = "drop"
  ) |>
  select(-year_month)

# Lag variances and compute unscaled portfolios
data_unscaled_full <- data_monthly_full |>
  mutate(
    mkt_excess_var_lag = lag(mkt_excess_var),
    smb_var_lag        = lag(smb_var),
    hml_var_lag        = lag(hml_var),
    rmw_var_lag        = lag(rmw_var),
    cma_var_lag        = lag(cma_var),
    mom_var_lag        = lag(mom_var),
    roe_var_lag        = lag(roe_var),
    ia_var_lag         = lag(ia_var),
    bab_var_lag        = lag(bab_var)
  ) |>
  filter(!is.na(mkt_excess_var_lag)) |>
  mutate(
    mkt_excess_unscaled = mkt_return_comp / mkt_excess_var_lag,
    smb_unscaled        = smb_return_comp  / smb_var_lag,
    hml_unscaled        = hml_return_comp  / hml_var_lag,
    rmw_unscaled        = rmw_return_comp  / rmw_var_lag,
    cma_unscaled        = cma_return_comp  / cma_var_lag,
    mom_unscaled        = mom_return_comp  / mom_var_lag,
    roe_unscaled        = roe_return_comp  / roe_var_lag,
    ia_unscaled         = ia_return_comp   / ia_var_lag,
    bab_unscaled        = bab_return_comp  / bab_var_lag
  )

# Scaling constants (full sample)
c_mkt_full <- sd(data_unscaled_full$mkt_return_comp, na.rm = TRUE) / sd(data_unscaled_full$mkt_excess_unscaled, na.rm = TRUE)
c_smb_full <- sd(data_unscaled_full$smb_return_comp, na.rm = TRUE) / sd(data_unscaled_full$smb_unscaled,        na.rm = TRUE)
c_hml_full <- sd(data_unscaled_full$hml_return_comp, na.rm = TRUE) / sd(data_unscaled_full$hml_unscaled,        na.rm = TRUE)
c_rmw_full <- sd(data_unscaled_full$rmw_return_comp, na.rm = TRUE) / sd(data_unscaled_full$rmw_unscaled,        na.rm = TRUE)
c_cma_full <- sd(data_unscaled_full$cma_return_comp, na.rm = TRUE) / sd(data_unscaled_full$cma_unscaled,        na.rm = TRUE)
c_mom_full <- sd(data_unscaled_full$mom_return_comp, na.rm = TRUE) / sd(data_unscaled_full$mom_unscaled,        na.rm = TRUE)
c_roe_full <- sd(data_unscaled_full$roe_return_comp, na.rm = TRUE) / sd(data_unscaled_full$roe_unscaled,        na.rm = TRUE)
c_ia_full  <- sd(data_unscaled_full$ia_return_comp,  na.rm = TRUE) / sd(data_unscaled_full$ia_unscaled,         na.rm = TRUE)
c_bab_full <- sd(data_unscaled_full$bab_return_comp, na.rm = TRUE) / sd(data_unscaled_full$bab_unscaled,        na.rm = TRUE)

# Apply scaling → full-sample managed portfolios
factors_vol_managed_full <- data_unscaled_full |>
  mutate(
    mkt_excess_managed = c_mkt_full * mkt_excess_unscaled,
    smb_managed        = c_smb_full * smb_unscaled,
    hml_managed        = c_hml_full * hml_unscaled,
    rmw_managed        = c_rmw_full * rmw_unscaled,
    cma_managed        = c_cma_full * cma_unscaled,
    mom_managed        = c_mom_full * mom_unscaled,
    roe_managed        = c_roe_full * roe_unscaled,
    ia_managed         = c_ia_full  * ia_unscaled,
    bab_managed        = c_bab_full * bab_unscaled,
    var_quintile = ntile(mkt_excess_var_lag, 5),
    var_quintile_labeled = factor(var_quintile, levels = 1:5,
                                  labels = c("Low Vol", "2", "3", "4", "High Vol"),
                                  ordered = TRUE)
  ) |>
  mutate(date_join = floor_date(date, "month")) |>
  left_join(nber_data, by = "date_join") |>
  select(-date_join)

# --- Figure 1b: Market factor — does the MM2017 pattern hold in extended sample? ---
quintile_summary_full <- factors_vol_managed_full |>
  group_by(var_quintile_labeled) |>
  summarize(
    mean_ret_orig_ann = mean(mkt_return_comp, na.rm = TRUE) * 12,
    sd_ret_orig_ann   = sd(mkt_return_comp,   na.rm = TRUE) * sqrt(12),
    mean_var_orig     = mean(mkt_return_comp, na.rm = TRUE) / var(mkt_return_comp, na.rm = TRUE),
    prob_recession    = mean(us_recession,    na.rm = TRUE),
    .groups = "drop"
  )

p1b_ret <- ggplot(quintile_summary_full, aes(x = var_quintile_labeled, y = mean_ret_orig_ann * 100)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  scale_y_continuous(limits = c(0, 12), breaks = seq(0, 12, 2)) +
  labs(title = "Average Return", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(panel.grid.major.x = element_blank())

p1b_sd <- ggplot(quintile_summary_full, aes(x = var_quintile_labeled, y = sd_ret_orig_ann * 100)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  scale_y_continuous(limits = c(0, 40), breaks = seq(0, 40, 10)) +
  labs(title = "Standard Deviation", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(panel.grid.major.x = element_blank())

p1b_ratio <- ggplot(quintile_summary_full, aes(x = var_quintile_labeled, y = mean_var_orig)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  scale_y_continuous(limits = c(0, 8), breaks = seq(0, 8, 2)) +
  labs(title = "E[R]/Var(R)", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(panel.grid.major.x = element_blank())

p1b_rec <- ggplot(quintile_summary_full, aes(x = var_quintile_labeled, y = prob_recession)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 0.5), breaks = seq(0, 0.5, 0.1)) +
  labs(title = "Probability of Recession", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(panel.grid.major.x = element_blank())

figure_1b <- (p1b_ret + p1b_sd) / (p1b_ratio + p1b_rec) +
  plot_annotation(
    title    = "Figure 1b: Market Factor — Extended Sample (1926–2025)",
    subtitle = "Replicates MM2017 Figure 1 on the full available sample"
  )
print(figure_1b)

# --- Figure 1b (comparison): Original vs Managed — extended sample ---
quintile_summary_full_comp <- factors_vol_managed_full |>
  group_by(var_quintile_labeled) |>
  summarize(
    mean_ret_orig_ann = mean(mkt_return_comp,   na.rm = TRUE) * 12,
    mean_ret_man_ann  = mean(mkt_excess_managed, na.rm = TRUE) * 12,
    sd_ret_orig_ann   = sd(mkt_return_comp,      na.rm = TRUE) * sqrt(12),
    sd_ret_man_ann    = sd(mkt_excess_managed,   na.rm = TRUE) * sqrt(12),
    mean_var_orig     = mean(mkt_return_comp,    na.rm = TRUE) / var(mkt_return_comp,    na.rm = TRUE),
    mean_var_man      = mean(mkt_excess_managed, na.rm = TRUE) / var(mkt_excess_managed, na.rm = TRUE),
    prob_recession    = mean(us_recession,        na.rm = TRUE),
    .groups = "drop"
  )

p1b_comp_ret <- quintile_summary_full_comp |>
  select(var_quintile_labeled, mean_ret_orig_ann, mean_ret_man_ann) |>
  pivot_longer(cols = -var_quintile_labeled, names_to = "portfolio", values_to = "return",
               names_prefix = "mean_ret_") |>
  mutate(portfolio = if_else(portfolio == "orig_ann", "Original", "Managed")) |>
  ggplot(aes(x = var_quintile_labeled, y = return, fill = portfolio)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = my_colors) +
  labs(title = "Average Return", x = "Variance Quintile", y = "Annualized Return", fill = "Portfolio") +
  theme_minimal() + theme(panel.grid.major.x = element_blank())

p1b_comp_sd <- quintile_summary_full_comp |>
  select(var_quintile_labeled, sd_ret_orig_ann, sd_ret_man_ann) |>
  pivot_longer(cols = -var_quintile_labeled, names_to = "portfolio", values_to = "std_dev",
               names_prefix = "sd_ret_") |>
  mutate(portfolio = if_else(portfolio == "orig_ann", "Original", "Managed")) |>
  ggplot(aes(x = var_quintile_labeled, y = std_dev, fill = portfolio)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = my_colors) +
  labs(title = "Standard Deviation", x = "Variance Quintile", y = "Annualized Std. Dev.", fill = "Portfolio") +
  theme_minimal() + theme(panel.grid.major.x = element_blank())

p1b_comp_ratio <- quintile_summary_full_comp |>
  select(var_quintile_labeled, mean_var_orig, mean_var_man) |>
  pivot_longer(cols = -var_quintile_labeled, names_to = "portfolio", values_to = "ratio",
               names_prefix = "mean_var_") |>
  mutate(portfolio = if_else(portfolio == "orig", "Original", "Managed")) |>
  ggplot(aes(x = var_quintile_labeled, y = ratio, fill = portfolio)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
  scale_fill_manual(values = my_colors) +
  labs(title = "E[R] / Var(R)", x = "Variance Quintile", y = "Ratio (Monthly)", fill = "Portfolio") +
  theme_minimal() + theme(panel.grid.major.x = element_blank())

p1b_comp_rec <- ggplot(quintile_summary_full_comp, aes(x = var_quintile_labeled, y = prob_recession)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "Probability of Recession", x = "Variance Quintile", y = "Probability") +
  theme_minimal() + theme(panel.grid.major.x = element_blank())

combined_plot_full <- (p1b_comp_ret + p1b_comp_sd) / (p1b_comp_ratio + p1b_comp_rec) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
combined_plot_full <- combined_plot_full +
  plot_annotation(
    title    = "Figure 1b (Comparison): Original vs. Managed Market — Extended Sample (1926–2025)",
    subtitle = "Market factor; variance quintiles based on lagged realized daily variance"
  )
print(combined_plot_full)

# --- Table 1b: Spanning regressions, all 9 factors, extended sample ---
# Reuses get_column_data() and factor_map defined in Part 1.
results_list_full <- list()
for (f_name in names(factor_map)) {
  cols <- factor_map[[f_name]]
  results_list_full[[f_name]] <- get_column_data(factors_vol_managed_full, cols[1], cols[2], f_name)
}

table_data_full <- bind_rows(results_list_full, .id = "column_factor") |>
  pivot_longer(cols = -column_factor) |>
  pivot_wider(names_from = column_factor, values_from = value) |>
  mutate(FX = NA_character_) |>
  add_row(name = "FX") |>
  relocate(FX, .after = CMA) |>
  mutate(
    FX = case_when(
      name == "FX" ~ "NA",
      name %in% c("Alpha", "N", "R2", "RMSE", "Alpha_FF3") ~ "NA",
      TRUE ~ FX
    )
  ) |>
  mutate(name = factor(name, levels = c(
    "MktRF", "SMB", "HML", "Mom", "RMW", "CMA", "FX", "ROE", "IA", "BAB",
    "Alpha", "N", "R2", "RMSE", "Alpha_FF3"
  ))) |>
  mutate(
    panel_group = ifelse(name == "Alpha_FF3",
                         "Panel B: Alphas controlling for FF3",
                         "Panel A: Univariate Regressions"),
    panel_group = factor(panel_group, levels = c("Panel A: Univariate Regressions",
                                                  "Panel B: Alphas controlling for FF3"))
  ) |>
  arrange(panel_group, name)

table_1b <- table_data_full |>
  gt(groupname_col = "panel_group") |>
  tab_header(title = "Table 1b: Extended Sample (1926–2025) — Replication of MM2017 Table I") |>
  cols_label(
    name  = "",
    MktRF = html("(1)<br>Mkt<sup>&sigma;</sup>"),
    SMB   = html("(2)<br>SMB<sup>&sigma;</sup>"),
    HML   = html("(3)<br>HML<sup>&sigma;</sup>"),
    Mom   = html("(4)<br>Mom<sup>&sigma;</sup>"),
    RMW   = html("(5)<br>RMW<sup>&sigma;</sup>"),
    CMA   = html("(6)<br>CMA<sup>&sigma;</sup>"),
    FX    = html("(7)<br>FX<sup>&sigma;</sup>"),
    ROE   = html("(8)<br>ROE<sup>&sigma;</sup>"),
    IA    = html("(9)<br>IA<sup>&sigma;</sup>"),
    BAB   = html("(10)<br>BAB<sup>&sigma;</sup>")
  ) |>
  text_transform(
    locations = cells_body(columns = name),
    fn = function(x) dplyr::case_when(x == "Alpha" ~ "Alpha (&alpha;)", x == "R2" ~ "R<sup>2</sup>", TRUE ~ x)
  ) |>
  fmt_markdown(columns = everything()) |>
  cols_align(align = "center", columns = -name) |>
  cols_align(align = "left",   columns = name) |>
  sub_missing(columns = everything(), missing_text = "") |>
  tab_options(
    table.border.top.color = "black", table.border.top.width = px(2),
    table.border.bottom.color = "black", table.border.bottom.width = px(2),
    heading.align = "center",
    column_labels.border.bottom.color = "black", column_labels.border.bottom.width = px(1),
    table.font.size = 12,
    row_group.font.weight = "bold", row_group.background.color = "#f0f0f0"
  )
print(table_1b)


############################ PART 2.  ##########################################

# Define start and end dates for analysis
start_date_part_2 <- ymd("1926-01-01") # Match factor data availability 
end_date_part_2 <- ymd("2025-07-31")   # Match factor data availability 

# --- 1. Load and Prepare Fama-French Data ---

factors_ff3_daily_raw_part_2 <- download_french_data("Fama/French 3 Factors [Daily]")

factors_ff3_daily_part_2 <- factors_ff3_daily_raw_part_2$subsets$data[[1]] |>
  mutate(
    date = ymd(date),
    across(c(RF, `Mkt-RF`, SMB, HML), ~as.numeric(.) / 100),
    .keep = "none"
  ) |>
  rename_with(str_to_lower) |>
  rename(mkt_excess = `mkt-rf`) |>
  filter(date >= start_date_part_2 & date <= end_date_part_2)

# For each month t, compute the realized variance using all daily returns within
# that month. Note that the number of trading days per month is not always the
# same

factors_ff3_monthly_part_2 <- factors_ff3_daily_part_2 |>
  mutate(
    year = year(date),
    month = month(date)
  ) |>
  group_by(year, month) |>
  summarize(
    date = max(date),

    mkt_excess_var = var(mkt_excess, na.rm = TRUE),
    smb_var = var(smb, na.rm = TRUE),
    hml_var = var(hml, na.rm = TRUE),
    
    .groups = "drop"
  ) |>
  arrange(date) |>
  select(date, mkt_excess_var, smb_var, hml_var)

unique(all_themes_monthly_vw_cap$location)

all_themes_monthly_vw_cap_wide_usa <- all_themes_monthly_vw_cap %>%
  mutate(date = ymd(date)) %>%
  filter(
    location == "usa",
    date >= start_date_part_2 & date <= end_date_part_2
  ) %>%
  select(date, name, ret) %>%
  pivot_wider(
    id_cols = date,
    names_from = name,
    values_from = ret,
    names_prefix = "ret_usa_"
  ) %>%
  arrange(date)

unique(all_themes_daily_vw_cap$location)

all_themes_daily_vw_cap_wide_usa <- all_themes_daily_vw_cap %>%
  mutate(date = ymd(date)) %>%
  filter(
    location == "usa",
    date >= start_date_part_2 & date <= end_date_part_2
  ) %>%
  select(date, name, ret) %>%
  pivot_wider(
    id_cols = date,
    names_from = name,
    values_from = ret,
    names_prefix = "ret_usa_"
  ) %>%
  arrange(date)

# --- Step 1: Setup ---
# 1. Define parameters based on student ID
digit_7 <- 2
digit_8 <- 6
D <- digit_7 + digit_8 + 10
print(paste("Using D =", D))

FACTOR_NAME <- "momentum" 
FACTOR_COL_NAME <- paste0("ret_usa_", FACTOR_NAME)

# 2. Update end date and load full Fama-French daily data
start_date_ext <- ymd("1926-07-01")
end_date_ext <- ymd("2025-07-31") # As per assignment 

# Get the full-sample daily market data
mkt_daily_ext <- factors_ff3_daily_raw_part_2$subsets$data[[1]] |>
  mutate(
    date = ymd(date),
    mkt_excess = as.numeric(`Mkt-RF`) / 100,
    rf = as.numeric(RF) / 100 
  ) |>
  select(date, mkt_excess, rf) |>
  filter(date >= start_date_ext & date <= end_date_ext)

# Get the daily factor data
if (!FACTOR_COL_NAME %in% names(all_themes_daily_vw_cap_wide_usa)) {
  stop(paste("Factor column", FACTOR_COL_NAME, "not found."))
}

factor_daily_ext <- all_themes_daily_vw_cap_wide_usa |>
  select(date, all_of(FACTOR_COL_NAME)) |>
  rename(factor_excess = !!FACTOR_COL_NAME) |>
  filter(date >= start_date_ext & date <= end_date_ext)

# 3. Join Mkt and Factor daily data
daily_data <- inner_join(mkt_daily_ext, factor_daily_ext, by = "date")

# --- Part 2: Step 1 ---

# Define the custom variance function to match the assignment's formula:
# formula: sigma_hat^2 = (D/22) * Sum_of_Squared_Deviations

calculate_assignment_var <- function(daily_returns_window) {
  
  # D_global is the target window size defined earlier.
  D_global <- D 
  
  # Get the number of non-missing observations in the current window
  n_valid_obs <- sum(!is.na(daily_returns_window))

  if (n_valid_obs < 2) {
    return(NA_real_)
  }
  
  # 1. Calculate the sample variance
  # s^2 = Sum_of_Squared_Deviations / (n - 1)
  s2 <- var(daily_returns_window, na.rm = TRUE)
  
  # 2. Back out the Sum of Squared Deviations (SSD)
  # SSD = s^2 * (n - 1)
  sum_sq_dev <- s2 * (n_valid_obs - 1)
  
  # 3. Apply the assignment's full formula: (D/22) * SSD
  # The scaling factor (D/22) uses the target window size, D_global
  assignment_var <- (D_global / 22) * sum_sq_dev
  
  return(assignment_var)
}


# 4. Calculate D-day rolling variance using the assignment's formula
daily_data_with_var <- daily_data |>
  arrange(date) |>
  tq_mutate(
    select = mkt_excess,
    mutate_fun = rollapply,
    width = D,
    FUN = calculate_assignment_var, 
    align = "right",
    fill = NA,
    col_rename = "mkt_d_day_var" 
  ) |>
  tq_mutate(
    select = factor_excess,
    mutate_fun = rollapply,
    width = D,
    FUN = calculate_assignment_var,
    align = "right",
    fill = NA,
    col_rename = "factor_d_day_var"
  ) |>
  filter(!is.na(mkt_d_day_var) & !is.na(factor_d_day_var))

# 5. Get end-of-month variance to match with next month's return
variance_monthly_lag <- daily_data_with_var |>
  mutate(year_month = floor_date(date, "month")) |>
  group_by(year_month) |>
  summarize(
    # Get the last D-day variance from that month
    mkt_var_lag = last(mkt_d_day_var),
    factor_var_lag = last(factor_d_day_var),
    .groups = "drop"
  ) |>
  mutate(
    # This variance from month 't' will be used for month 't+1' returns
    date = year_month + months(1)
  ) |>
  select(date, mkt_var_lag, factor_var_lag)


# --- Part 2: Step 2 ---
# 1. Load monthly Mkt returns (compounded from daily data)
mkt_monthly_ext <- daily_data |>
  mutate(
    mkt_return = mkt_excess + rf
  ) |>
  group_by(year_month = floor_date(date, "month")) |>
  summarize(
    mkt_return_comp = prod(1 + mkt_return) - 1,
    rf_comp = prod(1 + rf) - 1,
    mkt_return_comp = mkt_return_comp - rf_comp,
    .groups = "drop"
  ) |>
  rename(date = year_month) |> 
  select(date, mkt_return_comp)

# 2. Load monthly Factor returns
factor_monthly_ext <- all_themes_monthly_vw_cap_wide_usa |>
  select(date, all_of(FACTOR_COL_NAME)) |>
  rename(factor_excess_orig = !!FACTOR_COL_NAME) |>
  filter(date >= start_date_ext & date <= end_date_ext) |>
  mutate(date = floor_date(date, "month")) |> 
  select(date, factor_excess_orig)

# 3. Combine monthly returns and lagged variance
portfolios_unscaled <- inner_join(
  mkt_monthly_ext,
  factor_monthly_ext,
  by = "date"
) |>
  inner_join(
    variance_monthly_lag,
    by = "date"
  ) |>
  mutate(
    mkt_excess_unscaled = mkt_return_comp / mkt_var_lag,
    factor_excess_unscaled = factor_excess_orig / factor_var_lag
  ) |>
  filter(
    !is.na(mkt_excess_unscaled) & !is.na(factor_excess_unscaled) &
      !is.infinite(mkt_excess_unscaled) & !is.infinite(factor_excess_unscaled)
  )


# 4. Calculate scaling constants 'c' 
c_mkt <- sd(portfolios_unscaled$mkt_return_comp) / sd(portfolios_unscaled$mkt_excess_unscaled)
c_factor <- sd(portfolios_unscaled$factor_excess_orig) / sd(portfolios_unscaled$factor_excess_unscaled)

# 5. Create final scaled portfolios
portfolios_final <- portfolios_unscaled |>
  mutate(
    mkt_excess_managed = c_mkt * mkt_excess_unscaled,
    factor_excess_managed = c_factor * factor_excess_unscaled
  )

print("--- Part 2: Portfolio Scaling Verification ---")
print(paste("Market Scaling 'c':", round(c_mkt, 4)))
print(paste("Original Mkt SD:", round(sd(portfolios_final$mkt_return_comp), 6)))
print(paste("Managed Mkt SD: ", round(sd(portfolios_final$mkt_excess_managed), 6)))
print(paste("Factor Scaling 'c':", round(c_factor, 4)))
print(paste("Original Factor SD:", round(sd(portfolios_final$factor_excess_orig), 6)))
print(paste("Managed Factor SD: ", round(sd(portfolios_final$factor_excess_managed), 6)))




# --- Part 2: Step 3 (Calculate Metrics) ---

# 1. Convert our data to XTS objects
portfolios_matrix <- portfolios_final |>
  select(
    mkt_return_comp, mkt_excess_managed,
    factor_excess_orig, factor_excess_managed
  ) |>
  as.matrix()

portfolios_dates <- portfolios_final$date
portfolios_xts <- xts::xts(portfolios_matrix, order.by = portfolios_dates)

print("Successfully created XTS object:")
print(head(portfolios_xts))

# 2. Calculate Sharpe Ratios and Max Drawdowns
sharpe_ratios <- PerformanceAnalytics::SharpeRatio.annualized(portfolios_xts, scale = 12)
max_drawdowns <- PerformanceAnalytics::maxDrawdown(portfolios_xts)

# 3. Run regressions for Alphas
model_capm_man <- lm(mkt_excess_managed ~ mkt_return_comp, data = portfolios_final)
model_capm_factor_orig <- lm(factor_excess_orig ~ mkt_return_comp, data = portfolios_final)
model_capm_factor_man <- lm(factor_excess_managed ~ mkt_return_comp, data = portfolios_final)

# Alphas relative to their own original (unmanaged) factor
model_rel_mkt <- lm(mkt_excess_managed ~ mkt_return_comp, data = portfolios_final)
model_rel_factor <- lm(factor_excess_managed ~ factor_excess_orig, data = portfolios_final)

# Helper function
get_alpha_stats <- function(model) {
  # Use Newey-West (HAC) standard errors for robustness
  robust_test <- coeftest(model, vcov = vcovHAC(model))
  
  alpha_monthly_frac <- robust_test[1, "Estimate"]
  se_monthly_frac <- robust_test[1, "Std. Error"]
  t_stat <- robust_test[1, "t value"]
  
  alpha_ann_pct <- alpha_monthly_frac * 12 * 100
  
  return(
    sprintf("%.2f%% (t=%.2f)", alpha_ann_pct, t_stat)
  )
}

# 4. Assemble the final results table
results_summary <- tibble(
  Portfolio = c(
    "Market (Original)", "Market (Managed)",
    paste(FACTOR_NAME, "(Original)"), paste(FACTOR_NAME, "(Managed)")
  ),
  `Annualized Sharpe Ratio` = c(
    sharpe_ratios[,"mkt_return_comp"], sharpe_ratios[,"mkt_excess_managed"],
    sharpe_ratios[,"factor_excess_orig"], sharpe_ratios[,"factor_excess_managed"]
  ),
  `Max Drawdown` = c(
    max_drawdowns[,"mkt_return_comp"], max_drawdowns[,"mkt_excess_managed"],
    max_drawdowns[,"factor_excess_orig"], max_drawdowns[,"factor_excess_managed"]
  ),
  `CAPM Alpha (vs Mkt)` = c(
    "0.00% (t=0.00)",
    get_alpha_stats(model_capm_man),
    get_alpha_stats(model_capm_factor_orig),
    get_alpha_stats(model_capm_factor_man)
  ),
  `Alpha (vs Original)` = c(
    "---",
    get_alpha_stats(model_rel_mkt),
    "---",
    get_alpha_stats(model_rel_factor)
  )
)

# 5. Print the table using 'gt'
final_table <- results_summary |>
  mutate(
    `Annualized Sharpe Ratio` = round(as.numeric(`Annualized Sharpe Ratio`), 3),
    `Max Drawdown` = scales::percent(`Max Drawdown`, accuracy = 0.01)
  ) |>
  gt() |>
  tab_header(
    title = "Part 2: Volatility-Managed Portfolio Performance",
    subtitle = paste("Market vs.", str_to_title(FACTOR_NAME), "| D =", D, "days")
  ) |>
  cols_align(align = "center", columns = -Portfolio) |>
  cols_label(
    `Annualized Sharpe Ratio` = "Annual. Sharpe",
    `Max Drawdown` = "Max Drawdown",
    `CAPM Alpha (vs Mkt)` = "CAPM Alpha",
    `Alpha (vs Original)` = "Alpha vs. Original"
  ) |>
  tab_source_note(
    source_note = "Alphas are annualized (monthly alpha * 12) and in percent. t-statistics are from HAC robust standard errors."
  ) |>
  opt_table_outline()

print(final_table)


# --- Part 2: Step 4 (Scatter Plots) ---
cat("\nGenerating scatter plots for Part 2...\n")

scatter_p2_mkt <- ggplot(portfolios_final, aes(x = mkt_return_comp, y = mkt_excess_managed)) +
  geom_point(alpha = 0.4, color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
    title = "Part 2: Managed Market vs. Original Market Returns",
    x = "Original Market Excess Return",
    y = "Managed Market Return"
  ) +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

scatter_p2_factor <- ggplot(portfolios_final, aes(x = factor_excess_orig, y = factor_excess_managed)) +
  geom_point(alpha = 0.4, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
    title = paste("Part 2: Managed", str_to_title(FACTOR_NAME), "vs. Original Factor"),
    x = "Original Factor Excess Return",
    y = "Managed Factor Return"
  ) +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

# Combine and print scatter plots
scatter_plots_p2 <- scatter_p2_mkt + scatter_p2_factor
print(scatter_plots_p2)

# --- Part 2: Step 4 (Density Plots) ---

# --- 1: (1926-2015) ---

# Pivot the Part 1 data to a long format for ggplot
density_data_p1 <- factors_vol_managed %>%
  select(mkt_return_comp, mkt_excess_managed) %>%
  pivot_longer(
    cols = everything(),
    names_to = "portfolio_type",
    values_to = "returns"
  ) %>%
  mutate(
    portfolio_type = if_else(
      portfolio_type == "mkt_return_comp", 
      "Original", 
      "Managed"
    )
  )

# Plot the Part 1 densities
density_plot_p1 <- ggplot(density_data_p1, 
                          aes(x = returns, fill = portfolio_type)) +
  geom_density(alpha = 0.5) +
  
  scale_fill_manual(values = my_colors) +
  
  scale_x_continuous(
    labels = scales::percent_format(),
    limits = c(-0.3, 0.3)
  ) +
  labs(
    title = "Part 1: Return Distribution (1926-2015)",
    subtitle = "Managed portfolio has thinner tails and a higher peak.",
    x = "Monthly Excess Return",
    y = "Density",
    fill = "Portfolio Type"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(density_plot_p1)


# --- 2: Extended Sample ---

my_4_colors <- c(
  "Market (Original)" = "black",
  "Market (Managed)" = "#80bef1",
  "Momentum Factor (Original)" = "black",
  "Momentum Factor (Managed)" = "green4"
)

# Pivot the Part 2 data to a long format
density_data_p2 <- portfolios_final %>%
  select(mkt_return_comp, mkt_excess_managed, 
         factor_excess_orig, factor_excess_managed) %>%
  pivot_longer(
    cols = everything(),
    names_to = "portfolio_key",
    values_to = "returns"
  ) %>%
  mutate(
    Portfolio = case_when(
      grepl("mkt_", portfolio_key) ~ "Market",
      grepl("factor_", portfolio_key) ~ paste(str_to_title(FACTOR_NAME), "Factor")
    ),
    Type = case_when(
      grepl("_orig", portfolio_key) ~ "Original",
      grepl("_managed", portfolio_key) ~ "Managed"
    ),

    portfolio_fill = paste(Portfolio, Type, sep=" (") %>% paste0(")"),
    
    portfolio_fill = factor(portfolio_fill, levels = c(
      "Market (Original)",
      "Market (Managed)",
      "Momentum Factor (Original)",
      "Momentum Factor (Managed)"
    ))
  )

density_plot_p2 <- ggplot(density_data_p2,                        
                          aes(x = returns, fill = portfolio_fill)) +
  geom_density(alpha = 0.5) + 
  facet_wrap(~ Portfolio, scales = "free") + 
  scale_fill_manual(values = my_4_colors) +
  scale_x_continuous(labels = scales::percent_format()) +
  labs(
    title = "Return Distributions (Extended Sample)",
    subtitle = "Volatility management reduces tail risk in both portfolios.",
    x = "Monthly Excess Return",
    y = "Density",
    fill = "Portfolio"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(density_plot_p2)

# --- 2: Extended Sample (Separated Plots) ---

my_4_colors <- c(
  "Market (Original)" = "black",
  "Market (Managed)" = "#80bef1",
  "Momentum Factor (Original)" = "black",
  "Momentum Factor (Managed)" = "green4"
)

density_data_p2 <- portfolios_final %>%
  select(mkt_return_comp, mkt_excess_managed, 
         factor_excess_orig, factor_excess_managed) %>%
  pivot_longer(
    cols = everything(),
    names_to = "portfolio_key",
    values_to = "returns"
  ) %>%
  mutate(
    Portfolio = case_when(
      grepl("mkt_", portfolio_key) ~ "Market",
      grepl("factor_", portfolio_key) ~ paste(str_to_title(FACTOR_NAME), "Factor")
    ),
    Type = case_when(
      grepl("_orig", portfolio_key) ~ "Original",
      grepl("_managed", portfolio_key) ~ "Managed"
    ),
    
    portfolio_fill = paste(Portfolio, Type, sep=" (") %>% paste0(")"),
    
    portfolio_fill = factor(portfolio_fill, levels = c(
      "Market (Original)",
      "Market (Managed)",
      "Momentum Factor (Original)",
      "Momentum Factor (Managed)"
    ))
  )

density_plot_p2_mkt <- density_data_p2 %>%
  filter(Portfolio == "Market") %>%
  ggplot(aes(x = returns, fill = portfolio_fill)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = my_4_colors) + 
  scale_x_continuous(labels = scales::percent_format()) +
  labs(
    title = "Return Distribution: Market (Extended Sample)",
    subtitle = "Volatility management reduces tail risk.",
    x = "Monthly Excess Return",
    y = "Density",
    fill = "Portfolio"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

density_plot_p2_factor <- density_data_p2 %>%
  filter(Portfolio == "Momentum Factor") %>%
  ggplot(aes(x = returns, fill = portfolio_fill)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = my_4_colors) + 
  scale_x_continuous(labels = scales::percent_format()) +
  labs(
    title = "Return Distribution: Momentum Factor (Extended Sample)",
    subtitle = "Volatility management dramatically tames crash risk.",
    x = "Monthly Excess Return",
    y = "Density",
    fill = "Portfolio"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(density_plot_p2_mkt)
print(density_plot_p2_factor)

# --- Part 2: Step 4 (Cumulative Return Plot) ---

# Prepare data for cumulative plotting
plot_data_cumulative <- portfolios_final %>%
  select(date, 
         `Market (Original)` = mkt_return_comp, 
         `Market (Managed)` = mkt_excess_managed,
         `Momentum (Original)` = factor_excess_orig,
         `Momentum (Managed)` = factor_excess_managed
  ) %>%
  
  # Pivot to long format
  pivot_longer(
    cols = -date,
    names_to = "portfolio",
    values_to = "returns"
  ) %>%
  
  # Calculate cumulative (geometric) returns
  group_by(portfolio) %>%
  mutate(
    cumulative_return = cumprod(1 + returns) 
  ) %>%
  ungroup() %>%
  
  mutate(
    portfolio = factor(portfolio, levels = c(
      "Market (Original)", "Market (Managed)", 
      "Momentum (Original)", "Momentum (Managed)"
    ))
  )

cumulative_plot <- ggplot(plot_data_cumulative, 
                          aes(x = date, 
                              y = cumulative_return, 
                              color = portfolio)) +
  geom_line(linewidth=1) +
  # log scale for the y-axis
  scale_y_log10(
    labels = scales::dollar_format(prefix = "", suffix = "x") 
  ) +
  scale_color_manual(values = c(
    "Market (Original)" = "darkblue",
    "Market (Managed)" = "#80bef1",
    "Momentum (Original)" = "darkgreen",
    "Momentum (Managed)" = "green4"
  )) +
  
  labs(
    title = "Cumulative Excess Returns (1926 - 2025)",
    subtitle = "Volatility-Managed vs. Original Portfolios (Log Scale)",
    x = "Date",
    y = "Cumulative Return (Log Scale)",
    color = "Portfolio"
  ) +  
  theme_minimal() +
  theme(legend.position = "bottom")
print(cumulative_plot)


############################ PART 3: JKP Replication Quality Check ##########################################

# Install arrow if needed (required for reading parquet files)
if (!"arrow" %in% rownames(installed.packages())) install.packages("arrow")
library(arrow)

# --- Load JKP parquet files ---
thesis_weights <- read_parquet("data/processed/thesis_factor_weights.parquet") |>
  mutate(
    id  = as.numeric(id),   # arrow reads Int64 as bit64; cast to numeric for dplyr joins
    eom = as.Date(eom)
  )

monthly_ret <- read_parquet("data/processed/return_data/return_data/world_ret_monthly.parquet") |>
  filter(excntry == "USA") |>
  mutate(
    id  = as.numeric(id),
    eom = as.Date(eom)
  ) |>
  select(eom, id, ret_exc)

# --- Compute JKP factor returns: sum(weight_i * ret_exc_i) per month ---
# Weights at eom=t are formed using end-of-t characteristics and earn returns in t+1.
# Shift eom forward one month so the join matches weights(t) to ret_exc(t+1).
jkp_factors <- thesis_weights |>
  mutate(eom = eom %m+% months(1)) |>
  inner_join(monthly_ret, by = c("eom", "id")) |>
  group_by(eom) |>
  summarize(
    jkp_mktrf = sum(w_MktRF * ret_exc, na.rm = TRUE),
    jkp_smb   = sum(w_SMB   * ret_exc, na.rm = TRUE),
    jkp_hml   = sum(w_HML   * ret_exc, na.rm = TRUE),
    jkp_mom   = sum(w_MOM   * ret_exc, na.rm = TRUE),
    jkp_rmw   = sum(w_RMW   * ret_exc, na.rm = TRUE),
    jkp_cma   = sum(w_CMA   * ret_exc, na.rm = TRUE),
    jkp_roe   = sum(w_ROE   * ret_exc, na.rm = TRUE),
    jkp_ia    = sum(w_IA    * ret_exc, na.rm = TRUE),
    jkp_bab   = sum(w_BAB   * ret_exc, na.rm = TRUE),
    .groups = "drop"
  ) |>
  rename(date = eom)

# --- Get original monthly factor returns — extended sample from Part 1b (through 2025) ---
original_factors <- factors_vol_managed_full |>
  select(
    date,
    orig_mktrf = mkt_return_comp,
    orig_smb   = smb_return_comp,
    orig_hml   = hml_return_comp,
    orig_mom   = mom_return_comp,
    orig_rmw   = rmw_return_comp,
    orig_cma   = cma_return_comp,
    orig_roe   = roe_return_comp,
    orig_ia    = ia_return_comp,
    orig_bab   = bab_return_comp
  )

# --- Merge on common sample, starting 1964 (no end-date cap — use all available data) ---
comparison <- original_factors |>
  inner_join(jkp_factors, by = "date") |>
  filter(date >= ymd("1964-01-01"))

# --- Build one plot per factor ---
factor_specs <- list(
  list(label = "MktRF", orig = "orig_mktrf", jkp = "jkp_mktrf"),
  list(label = "SMB",   orig = "orig_smb",   jkp = "jkp_smb"),
  list(label = "HML",   orig = "orig_hml",   jkp = "jkp_hml"),
  list(label = "Mom",   orig = "orig_mom",   jkp = "jkp_mom"),
  list(label = "RMW",   orig = "orig_rmw",   jkp = "jkp_rmw"),
  list(label = "CMA",   orig = "orig_cma",   jkp = "jkp_cma"),
  list(label = "ROE",   orig = "orig_roe",   jkp = "jkp_roe"),
  list(label = "IA",    orig = "orig_ia",    jkp = "jkp_ia"),
  list(label = "BAB",   orig = "orig_bab",   jkp = "jkp_bab")
)

replication_plots <- lapply(factor_specs, function(fs) {
  corr <- cor(comparison[[fs$orig]], comparison[[fs$jkp]], use = "complete.obs")

  comparison |>
    select(date, Original = !!fs$orig, `JKP Replicated` = !!fs$jkp) |>
    filter(!is.na(Original) & !is.na(`JKP Replicated`)) |>
    arrange(date) |>
    mutate(
      Original        = cumprod(1 + Original),
      `JKP Replicated` = cumprod(1 + `JKP Replicated`)
    ) |>
    mutate(
      Original        = Original / first(Original),
      `JKP Replicated` = `JKP Replicated` / first(`JKP Replicated`)
    ) |>
    pivot_longer(cols = -date, names_to = "source", values_to = "cumret") |>
    ggplot(aes(x = date, y = cumret, color = source)) +
    geom_line(alpha = 0.75, linewidth = 0.4) +
    scale_color_manual(values = c("Original" = "darkblue", "JKP Replicated" = "#e05c2a")) +
    scale_y_log10(labels = scales::number_format(accuracy = 0.1, suffix = "x")) +
    labs(
      title    = fs$label,
      subtitle = sprintf("Correlation: %.3f", corr),
      x = NULL, y = "Cumulative Return (log scale)", color = NULL
    ) +
    theme_minimal(base_size = 9) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())
})

# --- Combine into 3x3 grid ---
replication_grid <- wrap_plots(replication_plots, ncol = 3) +
  plot_annotation(
    title    = "JKP Replicated vs. Original Factor Returns",
    subtitle = "Monthly returns over common sample | Orange = JKP, Blue = Original",
    theme = theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11)
    )
  ) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(replication_grid)

ggsave(
   filename = "factor_replication_comparison.png",
   plot     = replication_grid,                          
   width    = 16,
   height   = 14,
   dpi      = 150
 )


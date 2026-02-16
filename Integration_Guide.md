# Excel Portfolio Import - Integration Guide

## 📋 Overview

This guide shows you how to integrate Excel portfolio import functionality into your StockMind-AI tool. Users will be able to upload Excel files containing their portfolio data (ISIN, shares, price) and automatically sync it with the system.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Update requirements.txt

Add these lines to your `requirements.txt`:

```txt
openpyxl>=3.1.2          # Excel support
xlrd>=2.0.1              # Excel reading (legacy format support)
tenacity>=8.2.3          # Retry logic for robustness
```

Install:
```bash
pip install openpyxl xlrd tenacity
```

---

### Step 2: Add Functions to logic.py

Copy all functions from `excel_portfolio_import.py` and add them to your `logic.py` file. The key functions are:

- `read_excel_portfolio(file)` - Reads and parses Excel files
- `resolve_isin_to_ticker(isin)` - Converts ISIN codes to tickers
- `validate_portfolio_data(df)` - Validates imported data
- `sync_excel_portfolio(excel_df, merge_mode)` - Syncs with existing portfolio
- `generate_portfolio_template()` - Creates downloadable template
- `export_portfolio_to_excel(path)` - Exports current portfolio

**Where to add in logic.py:**
```python
# After the existing portfolio functions (around line 540)
# Add all the Excel import functions here
```

---

### Step 3: Update app.py Portfolio Tab

Replace the Portfolio tab section (Tab 5, around line 253) with the enhanced version from `app_portfolio_section.py`.

Don't forget to add this import at the top of `app.py`:
```python
import io  # For BytesIO operations
```

---

## 📊 Supported Excel Formats

### Format 1: Basic (Minimum Required)
```
| Ticker | Shares | Price |
|--------|--------|-------|
| AAPL   | 10     | 178.50|
| MSFT   | 5      | 420.75|
```

### Format 2: With Currency
```
| Ticker | Shares | Price  | Currency |
|--------|--------|--------|----------|
| AAPL   | 10     | 178.50 | USD      |
| VOD.L  | 100    | 73.50  | GBP      |
```

### Format 3: Full Details
```
| Ticker | Shares | Price  | Currency | Date       | Notes        |
|--------|--------|--------|----------|------------|--------------|
| AAPL   | 10     | 178.50 | USD      | 2024-01-15 | Tech stock   |
| MSFT   | 5      | 420.75 | USD      | 2024-02-01 | Cloud leader |
```

### Format 4: ISIN Codes (Auto-converted)
```
| ISIN         | Units | Price  | Currency |
|--------------|-------|--------|----------|
| US0378331005 | 10    | 178.50 | USD      |
| US5949181045 | 5     | 420.75 | USD      |
```

**Notes:**
- Column names are flexible (e.g., "Ticker" = "Symbol" = "Stock" = "ISIN")
- Spaces in column names are handled automatically
- Currency codes: USD, GBP, EUR, JPY, INR, GBX (pence)
- GBX (pence) is automatically converted to GBP
- Dates can be in various formats (YYYY-MM-DD, DD/MM/YYYY, etc.)

---

## 🎯 Import Modes

### 1. Add Mode (Default)
- Keeps all existing positions
- Adds new positions from Excel
- **Use when:** You want to supplement your portfolio

```python
# Example:
# Existing: AAPL (10 shares)
# Excel: MSFT (5 shares)
# Result: AAPL (10 shares) + MSFT (5 shares)
```

### 2. Update Mode
- Merges positions by ticker
- Averages prices if ticker exists
- Adds new tickers
- **Use when:** You want to update existing positions

```python
# Example:
# Existing: AAPL (10 shares @ $150)
# Excel: AAPL (5 shares @ $180)
# Result: AAPL (15 shares @ $160) [weighted average]
```

### 3. Replace Mode
- Closes all existing positions
- Imports fresh from Excel
- **Use when:** You want to start completely fresh

```python
# Example:
# Existing: AAPL, MSFT, GOOGL (all open)
# Excel: TSLA, NVDA
# Result: AAPL, MSFT, GOOGL (all closed), TSLA, NVDA (open)
```

---

## 🔍 Validation Features

The system automatically validates:

### ✅ What's Checked:
1. **Ticker existence** - Verifies ticker can fetch data
2. **Price sanity** - Checks for negative or extremely high prices
3. **Fractional shares** - Warns about unusual fractional shares
4. **Date validity** - Flags very old or future dates
5. **Data completeness** - Ensures required fields present

### ⚠️ Warnings vs Errors:

**Errors (Import blocked):**
- Invalid ticker that doesn't exist
- Negative or zero prices
- Missing required columns

**Warnings (Import allowed):**
- Unusually high prices (> $100,000)
- Fractional shares for stocks (common for ETFs/funds)
- Very old purchase dates (< year 2000)

---

## 💡 Usage Examples

### Example 1: Import from Broker Export

```python
# User downloads their broker's portfolio as Excel
# Uploads to StockMind-AI
# System automatically:
# 1. Reads all positions
# 2. Converts currencies to USD
# 3. Validates tickers
# 4. Shows preview
# 5. Imports on confirmation
```

### Example 2: ISIN-based Portfolio

```
| ISIN         | Quantity | Cost   | Currency |
|--------------|----------|--------|----------|
| US0378331005 | 10       | 178.50 | USD      | -> Resolves to AAPL
| GB0002374006 | 100      | 0.73   | GBP      | -> Resolves to VOD.L
```

The system:
1. Tries to find ticker using ISIN search
2. Falls back to company name search
3. Shows what it found for user verification

### Example 3: Multi-Currency Portfolio

```
| Ticker | Shares | Price  | Currency |
|--------|--------|--------|----------|
| AAPL   | 10     | 178.50 | USD      |
| VOD.L  | 100    | 73.50  | GBP      | -> Auto-converts to USD
| TCS.NS | 20     | 3500   | INR      | -> Auto-converts to USD
```

All prices stored in USD for consistent P&L calculations.

---

## 🛠️ Advanced Features

### Feature 1: Template Generator

```python
# In your app:
template_df = logic.generate_portfolio_template()
# Returns pre-filled example data
# User can download and modify
```

### Feature 2: Portfolio Export

```python
# Export current portfolio to Excel
logic.export_portfolio_to_excel("my_portfolio.xlsx")

# Excel includes:
# - All positions
# - Current prices
# - Unrealized P&L
# - P&L percentages
# - Formatted as table
# - Auto-sized columns
```

### Feature 3: Batch Validation

```python
validation = logic.validate_portfolio_data(excel_df)

# Returns:
{
    'valid': True/False,
    'issues': ['❌ BADTICK: Invalid ticker'],
    'warnings': ['⚠️ AAPL: High price $10000'],
    'total_positions': 10,
    'unique_tickers': 8,
    'total_value': 50000.00
}
```

---

## 🔒 Error Handling

### Common Issues & Solutions

**Issue 1: "Missing required columns"**
```
Solution: Ensure Excel has at minimum: Ticker/ISIN, Shares/Units, Price
Column names can vary but must contain these keywords
```

**Issue 2: "Could not fetch data for ticker"**
```
Solution: 
- Check ticker symbol is correct (e.g., AAPL not APPLE)
- For UK stocks, include .L suffix (e.g., VOD.L)
- For Indian stocks, use .NS or .BO suffix
- ISIN codes should work automatically
```

**Issue 3: "Invalid price"**
```
Solution:
- Remove currency symbols from price column (£, $, €)
- Use plain numbers: 178.50 not "$178.50"
- System will handle currency separately in Currency column
```

**Issue 4: GBX vs GBP confusion**
```
If your prices are in pence (GBX):
- Put "GBX" in Currency column
- System auto-converts to GBP by dividing by 100
```

---

## 📸 UI Screenshots (Text Description)

### Step 1: Template Download
```
┌─────────────────────────────────────┐
│ Download Excel Template             │
│ [📥 Download Excel Template]        │
└─────────────────────────────────────┘
```

### Step 2: Upload & Configure
```
┌─────────────────────────────────────┐
│ Upload Excel Portfolio              │
│ [Choose File: portfolio.xlsx]       │
│ Import Mode: [Add ▼]                │
└─────────────────────────────────────┘
```

### Step 3: Preview
```
┌─────────────────────────────────────┐
│ ✅ Loaded 5 positions from Excel    │
│                                     │
│ 📋 Preview Imported Data            │
│ ┌─────────────────────────────────┐│
│ │Ticker│Shares│Price│Currency│Date││
│ │AAPL  │  10  │178.5│  USD   │... ││
│ │MSFT  │   5  │420.7│  USD   │... ││
│ └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### Step 4: Validation
```
┌─────────────────────────────────────┐
│ Import Summary:                     │
│ - Total positions: 5                │
│ - Unique tickers: 5                 │
│ - Total cost basis: $10,451.25      │
│                                     │
│ [🚀 Confirm Import]                 │
└─────────────────────────────────────┘
```

---

## 🧪 Testing Checklist

Test these scenarios:

- [ ] Upload template file (should work perfectly)
- [ ] Upload with ISIN codes instead of tickers
- [ ] Upload with mixed currencies (USD, GBP, EUR)
- [ ] Upload with GBX (pence) - should convert to GBP
- [ ] Upload with missing optional columns (Currency, Date, Notes)
- [ ] Upload with invalid ticker (should show error)
- [ ] Upload with negative price (should show error)
- [ ] Test "Add" mode with existing portfolio
- [ ] Test "Update" mode (should average prices)
- [ ] Test "Replace" mode (should close existing)
- [ ] Download template - verify it's properly formatted
- [ ] Export portfolio - verify current data included

---

## 🐛 Troubleshooting

### Debug Mode

Add this to see what's happening:

```python
# In logic.py, at the top of read_excel_portfolio()
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Then add throughout the function:
logger.debug(f"Columns found: {df.columns.tolist()}")
logger.debug(f"Mapped columns: {actual_columns}")
logger.debug(f"Processing ticker: {ticker}")
```

### Common Fixes

```python
# If Excel reading fails:
# Try both engines:
df = pd.read_excel(file, engine='openpyxl')  # For .xlsx
df = pd.read_excel(file, engine='xlrd')      # For .xls (old format)

# If ISIN resolution fails:
# Check the search_ticker function is working:
results = search_ticker("US0378331005")
print(results)  # Should return ticker for Apple
```

---

## 📚 Full Code Locations

After integration, your files should look like:

```
stockmind-ai/
├── app.py                    # Streamlit UI (enhanced portfolio tab)
├── logic.py                  # Backend logic (with Excel functions)
├── Bot.py                    # GitHub bot (unchanged)
├── requirements.txt          # Dependencies (openpyxl added)
├── portfolio.csv             # Generated after imports
└── portfolio_template.xlsx   # Template for users
```

---

## 🎓 Next Steps

After Excel import is working:

1. **Add CSV support** - Simple extension of Excel reader
2. **Scheduled imports** - Auto-import from Dropbox/Google Drive
3. **Multi-portfolio support** - Different portfolios for different strategies
4. **Historical tracking** - Track portfolio value over time
5. **Tax reports** - Generate capital gains reports
6. **Broker integrations** - Direct API connections

---

## ❓ FAQ

**Q: Can I import fractional shares?**
A: Yes, the system supports any positive number including fractions (e.g., 0.5 shares).

**Q: What if my broker uses different column names?**
A: The system is flexible. It looks for keywords like "ticker", "symbol", "stock", "shares", "units", "quantity", etc.

**Q: Can I import multiple currencies?**
A: Yes! Just add a "Currency" column with codes (USD, GBP, EUR, etc.). The system converts everything to USD internally.

**Q: What happens to my existing positions?**
A: Depends on mode:
- "Add" - Keeps existing, adds new
- "Update" - Merges by ticker  
- "Replace" - Closes all, imports fresh

**Q: Do ISIN codes work for all stocks?**
A: The system tries to resolve ISINs but success depends on Yahoo Finance having the mapping. Ticker symbols are more reliable.

---

## 📞 Support

If you encounter issues:

1. Check the validation warnings/errors shown in UI
2. Try downloading and modifying the template
3. Verify ticker symbols on Yahoo Finance
4. Check logs for detailed error messages
5. Test with a small sample first (1-2 positions)

---

## ✅ Success Criteria

Your integration is complete when:

- ✅ Users can download Excel template
- ✅ Users can upload Excel with portfolio data
- ✅ System validates data and shows preview
- ✅ Import works in all three modes (Add/Update/Replace)
- ✅ ISIN codes are resolved to tickers
- ✅ Multiple currencies are converted to USD
- ✅ Validation catches invalid tickers/prices
- ✅ Users can export their current portfolio

---

Good luck with your integration! 🚀

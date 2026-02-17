# Excel Portfolio Import Enhancement for logic.py
# These functions extend logic.py with Excel import capabilities

import pandas as pd
import openpyxl
from typing import Optional, Dict, List
from logic import get_data, get_portfolio, search_ticker

PORTFOLIO_FILE = "portfolio.csv"

def read_excel_portfolio(file) -> Optional[pd.DataFrame]:
    """
    Read portfolio from Excel file with multiple format support.
    
    Expected columns (flexible naming):
    - ISIN / Ticker / Symbol
    - Shares / Units / Quantity
    - Price / Buy Price / Cost / Entry Price
    - Currency (optional, defaults to USD)
    - Date / Purchase Date (optional, defaults to today)
    - Notes (optional)
    
    Args:
        file: Uploaded Excel file object
    
    Returns:
        DataFrame with standardized columns or None if error
    """
    try:
        # Try reading Excel file
        df = pd.read_excel(file, engine='openpyxl')
        
        # Normalize column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Column mapping dictionary
        column_mappings = {
            'ticker': ['ticker', 'symbol', 'stock', 'code', 'isin'],
            'shares': ['shares', 'units', 'quantity', 'qty', 'amount'],
            'price': ['price', 'buy_price', 'cost', 'entry_price', 'purchase_price', 'avg_price'],
            'currency': ['currency', 'curr', 'ccy'],
            'date': ['date', 'purchase_date', 'buy_date', 'entry_date'],
            'notes': ['notes', 'note', 'comments', 'comment', 'description']
        }
        
        # Find actual column names
        actual_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                if col in possible_names:
                    actual_columns[standard_name] = col
                    break
        
        # Validate required columns
        required = ['ticker', 'shares', 'price']
        missing = [col for col in required if col not in actual_columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
        
        # Extract and standardize data
        portfolio_data = []
        
        for idx, row in df.iterrows():
            # Skip empty rows
            if pd.isna(row[actual_columns['ticker']]):
                continue
            
            ticker = str(row[actual_columns['ticker']]).strip().upper()
            
            # Handle ISIN codes - try to resolve to ticker
            if len(ticker) == 12 and ticker[:2].isalpha():  # ISIN format
                ticker = resolve_isin_to_ticker(ticker) or ticker
            
            # Parse shares
            try:
                shares = float(str(row[actual_columns['shares']]).replace(',', '').replace(' ', ''))
            except:
                continue  # Skip invalid shares
            
            # Parse price
            try:
                price_str = str(row[actual_columns['price']]).replace(',', '').replace(' ', '')
                # Remove currency symbols
                price_str = ''.join(c for c in price_str if c.isdigit() or c in '.-')
                price = float(price_str)
            except:
                continue  # Skip invalid price
            
            # Get currency (default USD)
            currency = "USD"
            if 'currency' in actual_columns:
                curr_val = str(row[actual_columns['currency']]).strip().upper()
                if curr_val in ['GBP', 'USD', 'EUR', 'JPY', 'INR', 'GBX']:
                    currency = curr_val
            
            # Convert GBX (pence) to GBP
            if currency == 'GBX':
                price = price / 100
                currency = 'GBP'
            
            # Get date (default today)
            buy_date = pd.Timestamp.now()
            if 'date' in actual_columns:
                try:
                    buy_date = pd.to_datetime(row[actual_columns['date']])
                except:
                    pass
            
            # Get notes
            notes = ""
            if 'notes' in actual_columns:
                notes = str(row[actual_columns['notes']]) if pd.notna(row[actual_columns['notes']]) else ""
            
            # Convert price to USD if needed
            price_usd = price
            if currency != 'USD':
                try:
                    rate = get_exchange_rate(currency)
                    price_usd = price * rate
                except:
                    # If conversion fails, use original price
                    pass
            
            portfolio_data.append({
                'Ticker': ticker,
                'Buy_Price_USD': price_usd,
                'Shares': shares,
                'Date': buy_date,
                'Status': 'OPEN',
                'Currency': currency,
                'Notes': notes
            })
        
        if not portfolio_data:
            raise ValueError("No valid portfolio entries found in Excel file")
        
        result_df = pd.DataFrame(portfolio_data)
        return result_df
        
    except Exception as e:
        print(f"Error reading Excel portfolio: {str(e)}")
        raise


def resolve_isin_to_ticker(isin: str) -> Optional[str]:
    """
    Try to resolve ISIN code to ticker symbol.
    
    Args:
        isin: ISIN code (e.g., US0378331005 for Apple)
    
    Returns:
        Ticker symbol or None if not found
    """
    try:
        # Search by ISIN
        results = search_ticker(isin)
        if results:
            # Return first result
            return list(results.values())[0]
    except:
        pass
    
    return None


def validate_portfolio_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate portfolio data and provide feedback.
    
    Args:
        df: Portfolio DataFrame
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        
        # Check if ticker exists
        data = get_data(ticker, period="1d", interval="1d")
        if data is None or data.empty:
            issues.append(f"⚠️ {ticker}: Could not fetch data - ticker may be invalid")
        
        # Check for unrealistic prices
        if row['Buy_Price_USD'] <= 0:
            issues.append(f"❌ {ticker}: Invalid price ${row['Buy_Price_USD']}")
        
        if row['Buy_Price_USD'] > 100000:
            warnings.append(f"⚠️ {ticker}: Unusually high price ${row['Buy_Price_USD']:,.2f}")
        
        # Check for fractional shares (unusual for stocks)
        if row['Shares'] % 1 != 0 and row['Shares'] < 10:
            warnings.append(f"⚠️ {ticker}: Fractional shares ({row['Shares']}) - is this an ETF?")
        
        # Check for very old dates
        if row['Date'] < pd.Timestamp('2000-01-01'):
            warnings.append(f"⚠️ {ticker}: Very old purchase date ({row['Date'].date()})")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'total_positions': len(df),
        'unique_tickers': df['Ticker'].nunique(),
        'total_value': (df['Buy_Price_USD'] * df['Shares']).sum()
    }


def sync_excel_portfolio(excel_df: pd.DataFrame, merge_mode: str = "add") -> bool:
    """
    Sync Excel portfolio with existing portfolio.
    
    Args:
        excel_df: DataFrame from Excel import
        merge_mode: 
            - "add": Add new positions without touching existing
            - "replace": Replace entire portfolio
            - "update": Update existing positions, add new ones
    
    Returns:
        True if successful
    """
    try:
        existing_df = get_portfolio()
        
        if merge_mode == "replace":
            # Close all existing positions first
            if not existing_df.empty:
                existing_df['Status'] = 'CLOSED'
                existing_df['Sell_Date'] = pd.Timestamp.now()
            
            # Add all Excel positions
            final_df = pd.concat([existing_df, excel_df], ignore_index=True)
        
        elif merge_mode == "update":
            # For each Excel entry, check if ticker already exists
            for _, excel_row in excel_df.iterrows():
                ticker = excel_row['Ticker']
                
                # Find existing open positions for this ticker
                existing_positions = existing_df[
                    (existing_df['Ticker'] == ticker) & 
                    (existing_df['Status'] == 'OPEN')
                ]
                
                if not existing_positions.empty:
                    # Update existing position (average price and sum shares)
                    idx = existing_positions.index[0]
                    old_shares = existing_df.at[idx, 'Shares']
                    old_price = existing_df.at[idx, 'Buy_Price_USD']
                    new_shares = excel_row['Shares']
                    new_price = excel_row['Buy_Price_USD']
                    
                    # Calculate weighted average price
                    total_shares = old_shares + new_shares
                    avg_price = (old_shares * old_price + new_shares * new_price) / total_shares
                    
                    existing_df.at[idx, 'Shares'] = total_shares
                    existing_df.at[idx, 'Buy_Price_USD'] = avg_price
                    existing_df.at[idx, 'Date'] = excel_row['Date']  # Update to latest date
                else:
                    # Add new position
                    existing_df = pd.concat([existing_df, pd.DataFrame([excel_row])], ignore_index=True)
            
            final_df = existing_df
        
        else:  # "add" mode (default)
            final_df = pd.concat([existing_df, excel_df], ignore_index=True)
        
        # Save to CSV
        final_df.to_csv(PORTFOLIO_FILE, index=False)
        return True
        
    except Exception as e:
        print(f"Error syncing portfolio: {str(e)}")
        return False


def generate_portfolio_template() -> pd.DataFrame:
    """
    Generate a sample Excel template for portfolio import.
    
    Returns:
        DataFrame with example data
    """
    template = pd.DataFrame([
        {
            'Ticker': 'AAPL',
            'Shares': 10,
            'Price': 178.50,
            'Currency': 'USD',
            'Date': '2024-01-15',
            'Notes': 'Example: Apple Inc'
        },
        {
            'Ticker': 'MSFT',
            'Shares': 5,
            'Price': 420.75,
            'Currency': 'USD',
            'Date': '2024-02-01',
            'Notes': 'Example: Microsoft'
        },
        {
            'Ticker': 'TSLA',
            'Shares': 3,
            'Price': 242.84,
            'Currency': 'USD',
            'Date': '2024-01-20',
            'Notes': 'Example: Tesla'
        }
    ])
    
    return template


def export_portfolio_to_excel(output_path: str = "portfolio_export.xlsx") -> bool:
    """
    Export current portfolio to Excel file.
    
    Args:
        output_path: File path for export
    
    Returns:
        True if successful
    """
    try:
        df = get_portfolio()
        
        if df.empty:
            return False
        
        # Add current value and P&L columns
        enhanced_df = df.copy()
        enhanced_df['Current_Price'] = 0.0
        enhanced_df['Current_Value'] = 0.0
        enhanced_df['Unrealized_PnL'] = 0.0
        enhanced_df['PnL_Pct'] = 0.0
        
        for idx, row in enhanced_df.iterrows():
            if row['Status'] == 'OPEN':
                data = get_data(row['Ticker'], period="1d", interval="1m")
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    enhanced_df.at[idx, 'Current_Price'] = current_price
                    enhanced_df.at[idx, 'Current_Value'] = current_price * row['Shares']
                    enhanced_df.at[idx, 'Unrealized_PnL'] = (current_price - row['Buy_Price_USD']) * row['Shares']
                    enhanced_df.at[idx, 'PnL_Pct'] = ((current_price / row['Buy_Price_USD']) - 1) * 100
        
        # Write to Excel with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            enhanced_df.to_excel(writer, sheet_name='Portfolio', index=False)
            
            # Get workbook and sheet
            workbook = writer.book
            worksheet = writer.sheets['Portfolio']
            
            # Format as table
            from openpyxl.worksheet.table import Table, TableStyleInfo
            
            table = Table(
                displayName="Portfolio",
                ref=f"A1:L{len(enhanced_df)+1}"
            )
            
            style = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False
            )
            
            table.tableStyleInfo = style
            worksheet.add_table(table)
            
            # Auto-size columns
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return True
        
    except Exception as e:
        print(f"Error exporting portfolio: {str(e)}")
        return False

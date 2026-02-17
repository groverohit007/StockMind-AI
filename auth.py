# auth.py
# Authentication Helper for Streamlit
# StockMind-AI Production

import streamlit as st
import database as db

def init_session_state():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = None
    if 'user_tier' not in st.session_state:
        st.session_state['user_tier'] = 'free'
    if 'is_admin' not in st.session_state:
        st.session_state['is_admin'] = False

def login_user(user_id, email, tier='free', is_admin=False):
    """Log in user."""
    st.session_state['logged_in'] = True
    st.session_state['user_id'] = user_id
    st.session_state['user_email'] = email
    st.session_state['user_tier'] = tier
    st.session_state['is_admin'] = is_admin

def logout_user():
    """Log out user."""
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['user_email'] = None
    st.session_state['user_tier'] = 'free'
    st.session_state['is_admin'] = False

def is_logged_in():
    """Check if user is logged in."""
    return st.session_state.get('logged_in', False)

def get_current_user_id():
    """Get current user ID."""
    return st.session_state.get('user_id')

def get_current_user_email():
    """Get current user email."""
    return st.session_state.get('user_email')

def is_premium():
    """Check if user is premium."""
    return st.session_state.get('user_tier') == 'premium'


def is_admin():
    """Check if current user has admin access."""
    return st.session_state.get('is_admin', False)

def show_login_page():
    """Display login/signup page."""
    st.title("🔐 StockMind-AI Login")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Welcome Back!")
        
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if email and password:
                    user_id = db.verify_user(email, password)
                    
                    if user_id:
                        user_info = db.get_user_info(user_id)
                        login_user(
                            user_id,
                            email,
                            user_info['subscription_tier'],
                            user_info.get('is_admin', False)
                        )
                        st.success("✅ Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
                else:
                    st.warning("⚠️ Please enter email and password")
    
    with tab2:
        st.subheader("Create Account")
        st.info("🎁 Get 2 free predictions per day!")
        
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
            
            agree = st.checkbox("I agree to Terms of Service and Privacy Policy")
            
            submit = st.form_submit_button("Create Account")
            
            if submit:
                if not agree:
                    st.warning("⚠️ Please agree to Terms of Service")
                elif not email or not password:
                    st.warning("⚠️ Please fill all fields")
                elif password != password_confirm:
                    st.error("❌ Passwords don't match")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    user_id = db.create_user(email, password)
                    
                    if user_id:
                        st.success("✅ Account created! Please log in.")
                        st.balloons()
                    else:
                        st.error("❌ Email already exists or error creating account")

def show_account_page():
    """Display user account page."""
    user_info = db.get_user_info(get_current_user_id())
    
    if not user_info:
        st.error("Error loading user information")
        return
    
    st.title("👤 My Account")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Account Info")
        st.write(f"**Email:** {user_info['email']}")
        st.write(f"**Member Since:** {user_info['created_at'][:10]}")
        st.write(f"**Subscription:** {user_info['subscription_tier'].title()}")
        st.write(f"**Status:** {user_info['subscription_status'].title()}")
    
    with col2:
        st.subheader("Usage Today")
        
        if user_info['subscription_tier'] == 'premium':
            st.success("✅ **Unlimited Predictions**")
        else:
            usage = db.get_usage_today(get_current_user_id())
            remaining = usage['limit'] - usage['used']
            
            st.write(f"**Used:** {usage['used']}/{usage['limit']}")
            st.write(f"**Remaining:** {remaining}")
            
            if remaining == 0:
                st.warning("⚠️ Daily limit reached!")
                st.info("💎 Upgrade to Premium for unlimited predictions!")
    
    # Logout button
    st.markdown("---")
    if st.button("🚪 Logout", type="secondary"):
        logout_user()
        st.rerun()

def require_login(func):
    """Decorator to require login for a function."""
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            show_login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def check_prediction_limit():
    """Check if user can make prediction and show appropriate message."""
    user_id = get_current_user_id()
    
    if is_premium():
        return True
    
    if db.can_make_prediction(user_id):
        remaining = db.get_remaining_predictions(user_id)
        st.info(f"ℹ️ {remaining} free predictions remaining today")
        return True
    else:
        st.error("❌ Daily prediction limit reached!")
        st.warning("⏰ Limit resets at midnight UTC")
        
        st.markdown("---")
        st.subheader("💎 Upgrade to Premium")
        st.write("**Premium Benefits:**")
        st.write("• ✅ Unlimited predictions")
        st.write("• ✅ All 4 timeframes (hourly, daily, weekly, monthly)")
        st.write("• ✅ Priority support")
        st.write("• ✅ Advanced features")
        st.write("")
        st.write("**Only £17/month**")
        
        return False

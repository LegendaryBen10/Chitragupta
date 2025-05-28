import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
sys.path.insert(0, '/usr/lib/chromium-browser/chromedriver') # Specific to your environment
import os
os.environ['DISPLAY'] = ':0' # Specific to your environment
import chromadb
from chromadb.config import Settings
# import requests # Not used directly in app.py's core logic anymore for Gemini
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# --- New imports for Queueing System ---
import queue
import threading
import uuid
# --- End New imports ---

# Initialize Flask app after imports
app = Flask(__name__)
CORS(app)

# Initialize ChromaDB client
# Ensure Settings are appropriate if you persist ChromaDB
client = chromadb.Client(Settings())

# --- Global components for the Queueing System ---
request_queue = queue.Queue()  # Thread-safe queue to hold incoming tasks
results_store = {}             # Dictionary to store results: {request_id: (response_text, context_text)}
result_events = {}             # Dictionary to store threading.Event objects: {request_id: event_object}
shared_resource_lock = threading.Lock() # Lock for secure access to shared dictionaries

# --- Gemini API Key Configuration ---
# IMPORTANT: Replace these with your actual API keys, preferably loaded from environment variables
GEMINI_API_KEYS = [
    "AIzaSyCjPOOS0s8Uw3f6RzUXpMYUziTMINFtkMs",
    "AIzaSyCuy_YHMwDlQmG0_rfPKNIQxUTQ1HB_-pE",
    # Add more keys if available
]
if not GEMINI_API_KEYS or GEMINI_API_KEYS[0] == "YOUR_GEMINI_API_KEY_1":
    print("WARNING: Update GEMINI_API_KEYS in app.py with your actual API keys.")
    # Fallback to a single key from original code if not configured, but rotation won't work.
    # Consider raising an error or using a default non-functional key to force configuration.
    GEMINI_API_KEYS = ["AIzaSyDICG9LCC31Im5u72t4CK1Wp9ByA8zIWRs"] # Original key as fallback

current_api_key_index = 0
# --- End Gemini API Key Configuration ---


# Webscraping - Table data - Generalized function
def scrape_table_from_page(url, click_sequence, table_selector, headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    try:
        wait = WebDriverWait(driver, 15)
        for selector in click_sequence:
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, table_selector)))
        time.sleep(2)
        print(f"✅ Table loaded from {url}!")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.select_one(table_selector)
        if table:
            headings = [th.get_text(strip=True) for th in table.find_all("th")]
            data = []
            for row in table.find_all("tr"):
                cols = [col.get_text(strip=True) for col in row.find_all("td")]
                if cols:
                    row_data = {headings[i]: cols[i] for i in range(len(cols)) if i < len(headings)}
                    data.append(row_data)
            return json.dumps(data, indent=4)
        else:
            print(f"⚠️ Table not found at {url} with selector {table_selector}!")
            return None
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None
    finally:
        driver.quit()
    
import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_headers_paragraphs_by_selectors(url, click_sequence, header_selectors, paragraph_selectors, headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 15)

        for selector in click_sequence:
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()
            time.sleep(1)

        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        if len(header_selectors) != len(paragraph_selectors):
            raise ValueError("❌ header_selectors and paragraph_selectors must have the same length")

        content_map = {}

        for i in range(len(header_selectors)):
            header_el = soup.select_one(header_selectors[i])
            header_text = header_el.get_text(strip=True) if header_el else f"[Missing header {i}]"

            para_texts = []
            for para_sel in paragraph_selectors[i]:
                para_el = soup.select_one(para_sel)
                if para_el:
                    para_texts.append(para_el.get_text(strip=True))
                else:
                    para_texts.append("[Missing paragraph]")

            combined_para = " ".join(para_texts)
            content_map[header_text] = combined_para

        return json.dumps(content_map, indent=4)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        driver.quit()
        

import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_customheaders_paragraphs_by_selectors(url, click_sequence, header_selectors, paragraph_selectors, custom_headers=None, headless=True):
    """
    Scrape paragraphs with optional custom headers.

    Args:
        url (str): URL of the page to scrape.
        click_sequence (list): CSS selectors to click in sequence.
        header_selectors (list): CSS selectors for headers.
        paragraph_selectors (list of lists): List of CSS selectors for paragraphs per header.
        headless (bool): Headless browser mode.
        custom_headers (list or None): Optional custom headers.

    Returns:
        str: JSON-formatted string with scraped data.
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 15)

        for selector in click_sequence:
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()
            time.sleep(1)

        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        if len(header_selectors) != len(paragraph_selectors):
            raise ValueError("❌ header_selectors and paragraph_selectors must have the same length")

        if custom_headers and len(custom_headers) != len(paragraph_selectors):
            raise ValueError("❌ custom_headers must have the same length as paragraph_selectors")

        content_map = {}

        for i in range(len(header_selectors)):
            # Use custom header if provided
            if custom_headers and custom_headers[i]:
                header_text = custom_headers[i]
            else:
                header_el = soup.select_one(header_selectors[i])
                header_text = header_el.get_text(strip=True) if header_el else f"[Missing header {i}]"

            para_texts = []
            for para_sel in paragraph_selectors[i]:
                para_el = soup.select_one(para_sel)
                if para_el:
                    para_texts.append(para_el.get_text(strip=True))
                else:
                    para_texts.append("[Missing paragraph]")

            combined_para = " ".join(para_texts)
            content_map[header_text] = combined_para

        return json.dumps(content_map, indent=4)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        driver.quit()
        

import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_headers_tables_by_selectors(url, click_sequence, header_selectors, table_selectors, headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 15)

        for selector in click_sequence:
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()
            time.sleep(1)

        time.sleep(2)  # Allow extra time for dynamic loading

        soup = BeautifulSoup(driver.page_source, "html.parser")

        if len(header_selectors) != len(table_selectors):
            raise ValueError("❌ header_selectors and table_selectors must have the same length")

        content_map = {}

        for i in range(len(header_selectors)):
            # Extract header text
            header_el = soup.select_one(header_selectors[i])
            header_text = header_el.get_text(strip=True) if header_el else f"[Missing header {i}]"

            # Extract table data
            table_el = soup.select_one(table_selectors[i])
            if table_el:
                headings = [th.get_text(strip=True) for th in table_el.find_all("th")]
                data_rows = []
                for row in table_el.find_all("tr"):
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cols:
                        row_data = {headings[j]: cols[j] for j in range(len(cols)) if j < len(headings)}
                        data_rows.append(row_data)
                content_map[header_text] = data_rows
            else:
                content_map[header_text] = f"[Missing table {i}]"

        return json.dumps(content_map, indent=4)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        driver.quit()
        

import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_customheaders_tables_by_selectors(url, click_sequence, header_selectors, table_selectors,custom_headers=None, headless=True):
    """
    Scrape tables and headers from a web page, with optional custom headers.

    Args:
        url (str): The URL to scrape.
        click_sequence (list): List of CSS selectors to click to navigate to content.
        header_selectors (list): List of CSS selectors for the table headers.
        table_selectors (list): List of CSS selectors for the tables.
        headless (bool): Run browser in headless mode.
        custom_headers (list or None): Optional custom header texts for the tables.

    Returns:
        str: JSON-formatted string of extracted data.
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 15)

        for selector in click_sequence:
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()
            time.sleep(1)

        time.sleep(2)  # Allow extra time for dynamic loading

        soup = BeautifulSoup(driver.page_source, "html.parser")

        if len(header_selectors) != len(table_selectors):
            raise ValueError("❌ header_selectors and table_selectors must have the same length")

        if custom_headers and len(custom_headers) != len(table_selectors):
            raise ValueError("❌ custom_headers must have the same length as table_selectors")

        content_map = {}

        for i in range(len(header_selectors)):
            # Use custom header if provided
            if custom_headers and custom_headers[i]:
                header_text = custom_headers[i]
            else:
                header_el = soup.select_one(header_selectors[i])
                header_text = header_el.get_text(strip=True) if header_el else f"[Missing header {i}]"

            # Extract table data
            table_el = soup.select_one(table_selectors[i])
            if table_el:
                headings = [th.get_text(strip=True) for th in table_el.find_all("th")]
                data_rows = []
                for row in table_el.find_all("tr"):
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cols:
                        row_data = {headings[j]: cols[j] for j in range(len(cols)) if j < len(headings)}
                        data_rows.append(row_data)
                content_map[header_text] = data_rows
            else:
                content_map[header_text] = f"[Missing table {i}]"

        return json.dumps(content_map, indent=4)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        driver.quit()
        

def upload_to_chromadb(json_string, name, desc="", collection_name="kmit"):
    if not json_string: # Check if json_string is None or empty
        print(f"⚠️ Skipping upload for '{name}' to ChromaDB as data is empty.")
        return
    if not isinstance(json_string, str):
        print(f"❌ json_string for '{name}' must be a string, got {type(json_string)}")
        return
    if not isinstance(name, str):
        print("❌ name must be a string")
        return

    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    metadata = {"description": desc} if desc else {}
    collection.add(
        documents=[json_string],
        metadatas=[metadata],
        ids=[name]
    )
    print(f"✅ Document '{name}' uploaded to ChromaDB collection '{collection_name}'")

def query_chroma(collection, query):
    results = collection.query(query_texts=[query], n_results=5)
    # Ensure results are not empty and documents exist
    if results and results['documents'] and results['documents'][0]:
        return results['documents'][0] # Retrieve the most relevant document list (which contains strings)
    return "" # Return empty string if no relevant documents found

def generate_response_with_gemini(context, query, api_key):
    # Configure Gemini with the specific API key for this call
    genai.configure(api_key=api_key)
    prompt = f"""
    Your name is Chitraguptha.
    Answer in a polite and detailed manner like you are a virtual assistant for the KMIT college website.
    Kmit website data:{context}
    Answer the query:{query}
    Please answer in a full sentence and in plain text.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite") # Using gemini-1.5-pro
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API Error: {e} (using key ending ...{api_key[-4:] if api_key else 'N/A'})")
        # If a specific error indicates a rate limit, we might want to signal that
        # for more intelligent key rotation, but for now, just return an error message.
        if "API key not valid" in str(e) or "permission" in str(e).lower():
            raise # Re-raise specific key-related errors to be handled by worker's rotation
        return f"Sorry, I encountered an issue while trying to generate a response. Error: {e}"


def run_query_for_worker(collection, query, api_key_to_use):
    """
    This function will be called by the worker.
    It includes the ChromaDB query and the Gemini API call.
    """
    retrieved_data_list = query_chroma(collection, query) # This is a list of strings
    
    # Join the list of document strings into a single context string
    context = " ".join(retrieved_data_list) if isinstance(retrieved_data_list, list) else retrieved_data_list
    
    if not context:
      print(f"⚠️ No context found in ChromaDB for query: {query}")
      # Decide if you want to proceed to Gemini without context or return a specific message
      # context = "No specific information found in the college database for this query."

    # print(f"Worker CONTEXT for query '{query}': \n{context[:200]}...") # Print start of context

    response_text = generate_response_with_gemini(context, query, api_key_to_use)
    # print(f"\nWorker 🔍 Query: {query}\nWorker ✅ Response: {response_text}\n")
    return response_text, context


# --- Background Worker Function for Gemini API Calls---
def gemini_api_call_worker():
    global current_api_key_index
    global GEMINI_API_KEYS

    print("INFO: Gemini API call worker thread started.")
    kmit_collection = None
    try:
        kmit_collection = client.get_collection(name="kmit")
    except Exception as e:
        print(f"CRITICAL: Worker could not get ChromaDB collection 'kmit'. Error: {e}. Worker stopping.")
        return # Stop worker if DB is not accessible

    while True:
        task_data = None
        request_id = None
        original_thread_event = None
        try:
            task_data = request_queue.get() # Blocks until an item is available
            request_id = task_data['id']
            user_query = task_data['query']
            original_thread_event = task_data['event']

            print(f"WORKER: Processing request_id: {request_id} for query: '{user_query}'")

            api_key_to_use = None
            response_text = None
            context_text = ""
            max_retries_per_key = 2 # How many times to retry with the same key for generic errors
            total_key_rotations = 0 # How many times we've tried all keys

            while total_key_rotations < len(GEMINI_API_KEYS) * 2: # Try each key up to twice
                with shared_resource_lock:
                    current_api_key_index = (current_api_key_index % len(GEMINI_API_KEYS)) # Ensure index is valid
                    api_key_to_use = GEMINI_API_KEYS[current_api_key_index]
                
                print(f"WORKER: Attempting query for {request_id} with API Key ending ...{api_key_to_use[-4:]}")
                
                try:
                    # Call the combined function
                    response_text, context_text = run_query_for_worker(kmit_collection, user_query, api_key_to_use)
                    
                    # Check if response indicates an internal Gemini error passed as text
                    if "Sorry, I encountered an issue" in response_text and "API key not valid" not in response_text:
                        # This might be a model issue, not a key issue. Break to not rotate uselessly.
                        print(f"WORKER: Gemini returned an issue for {request_id}, not necessarily key-related. Using response.")
                        break 
                    elif "API key not valid" in response_text or "permission" in str(response_text).lower(): # Assuming generate_response_with_gemini returns this string
                        print(f"WORKER: API Key ...{api_key_to_use[-4:]} seems invalid or has permission issues for {request_id}.")
                        raise ValueError("Simulated API Key Error for rotation") # Force rotation

                    print(f"WORKER: Successfully processed {request_id} with key ...{api_key_to_use[-4:]}")
                    break # Success, exit retry loop

                except ValueError as ve: # Catch our simulated key error or actual key errors re-raised
                    print(f"WORKER: Key ...{api_key_to_use[-4:]} failed for {request_id}. Rotating. Error: {ve}")
                    with shared_resource_lock:
                        current_api_key_index = (current_api_key_index + 1) % len(GEMINI_API_KEYS)
                    total_key_rotations += 1
                    time.sleep(1) # Brief pause before trying next key
                    if response_text is None: # Ensure response_text is not None if all keys fail
                        response_text = "Sorry, all attempts to contact the AI service failed due to key or permission issues."


                except Exception as e:
                    print(f"WORKER: Error during Gemini call for {request_id} with key ...{api_key_to_use[-4:]}. Error: {e}")
                    # For other errors, you might retry with the same key or rotate.
                    # For simplicity here, we'll rotate.
                    with shared_resource_lock:
                        current_api_key_index = (current_api_key_index + 1) % len(GEMINI_API_KEYS)
                    total_key_rotations +=1
                    time.sleep(1)
                    if response_text is None:
                        response_text = f"An unexpected error occurred while processing your query. Error: {e}"
            
            if response_text is None: # Should be set by loops above
                 response_text = "Apologies, the AI service could not process your request at this time."


            print(f"WORKER: Final response for {request_id}: '{response_text[:100]}...'")
            with shared_resource_lock:
                results_store[request_id] = (response_text, context_text) # Store as tuple
            original_thread_event.set()
            request_queue.task_done()

        except Exception as e_outer:
            print(f"CRITICAL WORKER ERROR: {e_outer} for task_data: {task_data}")
            if request_id and original_thread_event and not original_thread_event.is_set():
                with shared_resource_lock:
                    results_store[request_id] = (f"A critical error occurred in the processing worker: {e_outer}", "")
                original_thread_event.set() # Ensure Flask route doesn't hang indefinitely
# --- End Background Worker ---


# --- Flask Route for Queries ---
@app.route('/query', methods=['POST'])
def process_query_endpoint():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        user_query = data['query']
        request_id = str(uuid.uuid4())
        event_for_this_request = threading.Event()

        task_to_queue = {
            'id': request_id,
            'query': user_query,
            'event': event_for_this_request
        }
        request_queue.put(task_to_queue)
        print(f"FLASK_ROUTE: Queued request {request_id} for query: '{user_query}'")

        # Wait for the worker to process this request, with a timeout
        event_set = event_for_this_request.wait(timeout=120.0) # Increased timeout

        if not event_set:
            print(f"FLASK_ROUTE: Timeout for request {request_id}")
            # Attempt to clean up, though worker might still complete and log an error if it tries to set results
            with shared_resource_lock:
                results_store.pop(request_id, None)
            return jsonify({'error': 'Processing your request timed out. Please try again.'}), 504

        with shared_resource_lock:
            response_tuple = results_store.pop(request_id, ("Error: Result not found in store.", ""))
        
        final_response_text, _ = response_tuple # context_text is also available if needed

        print(f"FLASK_ROUTE: Responding for request {request_id}")
        return jsonify({
            'query': user_query,
            'response': final_response_text,
            # 'context': context_text # You can uncomment this if frontend needs it
        })

    except Exception as e:
        print(f"FLASK_ROUTE_ERROR: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500
# --- End Flask Route ---

# --- Main Execution Block ---
if __name__ == '__main__':
    print("INFO: Starting application setup...")

    # SCRAPING DATA FROM KMIT WEBSITE (runs once at startup)
    print("INFO: Starting data scraping...")
    # (Your scraping calls - ensure they handle None results gracefully for ChromaDB upload)
    # Example:
    #director academic
    diraca_url = "https://kmit.in/administration/academicdirector.php"
    clicks = []

    headers = [
        ""

    ]

    custom = [
        "Director of Academics"
    ]

    paragraphs = [
        ["body > div > section > div > div.row > div.col-sm-9 > p"]

    ]

    diraca_data = scrape_customheaders_paragraphs_by_selectors(diraca_url, clicks, headers, paragraphs,custom_headers=custom)

    #24-25 placements
    latest_placement_url = "https://kmit.in/placements/placement.php"
    clicks = ["body > div.background > div:nth-child(2) > div > ul > li:nth-child(4) > a"]
    headers=[""]
    custom = [
        "2024-2025(2024-25) placements table"
    ]
    latest_table_selector = ["#cp2024-25 > div > table"]
    placements_data = scrape_customheaders_tables_by_selectors(latest_placement_url, clicks, headers, latest_table_selector,custom_headers=custom)
    # print(placements_data)

    #23-24 placements
    old_placement_url = "https://kmit.in/placements/placement.php"
    clicks = [
        "body > div.background > div:nth-child(2) > div > ul > li:nth-child(4) > a",
        "#campus > div > ul > li:nth-child(2) > a > b"
    ]
    headers=[""]
    custom = [
        "2023-2024(2023-24) placements table"
    ]
    old_table_selector = ["#cp2023-24 > div > table"]

    placements_data_2023_2024 = scrape_customheaders_tables_by_selectors(old_placement_url, clicks, headers, old_table_selector,custom_headers=custom)
    #print(placements_data_2023_2024)

    #admission procedure
    #(with headers)
    admission_url = "https://kmit.in/admissions/admission-procedure.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#kmitra > div > div > div > h5:nth-child(10)"]
    admission_table_selector = ["#kmitra > div > div > div > div > div.box > table"]
    admissions_data = scrape_headers_tables_by_selectors(admission_url, clicks, headers, admission_table_selector)
    #print(admissions_data)

    # #(without headers)
    # admission_url = "https://kmit.in/admissions/admission-procedure.php"
    # clicks = []  # No clicks needed for admissions table
    # admission_table_selector = "div.box table.table.table-striped.custom"
    # admissions_data = scrape_table_from_page(admission_url, clicks, admission_table_selector)
    # #print(admissions_data)

    #couses offered
    courses_url="https://kmit.in/admissions/coursesoffered.php"
    clicks = []  # No clicks needed for admissions table
    courses_table_selector = "div.box table.table.table-striped.custom"
    courses_data = scrape_table_from_page(courses_url, clicks, courses_table_selector)

    #cse faculty
    cse_url="https://www.kmit.in/department/faculty_CSE.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#Sponsored\ Research > div:nth-child(1) > header > h4"]
    cse_table_selector = ["div.box table.table.table-striped.custom"]
    cse_data = scrape_headers_tables_by_selectors(cse_url, clicks , headers , cse_table_selector)
    #print(cse_data)

    #csm faculty
    csm_url="https://www.kmit.in/department/faculty_csm.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#faculty-CSM\(AI\&ML\) > div.container > div > div:nth-child(1) > header > h4"]
    csm_table_selector = ["#faculty-CSM\(AI\&ML\) > div.container > div > div:nth-child(2) > div > div > table"]
    csm_data = scrape_headers_tables_by_selectors(csm_url, clicks , headers , csm_table_selector)
    #print(csm_data)

    #it faculty
    it_url="https://www.kmit.in/department/faculty_it.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#faculty-IT > div.container > div > div:nth-child(1) > header > h4"]
    it_table_selector = ["div.box table.table.table-striped.custom"]
    it_data = scrape_headers_tables_by_selectors(it_url, clicks , headers , it_table_selector)
    #print(it_data)

    #csd faculty
    csd_url="https://www.kmit.in/department/faculty_csd.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#faculty-CSE\(DATA\ SCIENCE\) > div.container > div > div:nth-child(1) > header > h4"]
    csd_table_selector = ["div.box table.table.table-striped.custom"]
    csd_data = scrape_headers_tables_by_selectors(csd_url, clicks , headers , csd_table_selector)
    #print(csd_data)

    # #hs faculty
    hs_url="https://www.kmit.in/department/faculty_hs.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#faculty-H\&S > div > div.row > div > header > h4"]
    hs_table_selector = ["div.box table.table.table-striped.custom"]
    hs_data = scrape_headers_tables_by_selectors(hs_url, clicks, headers, hs_table_selector)
    #print(hs_data)

    # #research publications
    research_url="https://kmit.in/research/researchpublications.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#researchpublications > div > header > h4"]
    research_table_selector = ["div.box table.table-striped"]
    research_data = scrape_headers_tables_by_selectors(research_url, clicks, headers, research_table_selector)
    #print(research_data)

    # #contact exam
    contact_url="https://kmit.in/examination/contact_exam.php"
    clicks = []  # No clicks needed for admissions table
    contact_table_selector = "table"
    contact_data = scrape_table_from_page(contact_url, clicks, contact_table_selector)
    #print(contact_data)

    #student council
    council_url="https://kmit.in/intiatives/studentcouncil.php"
    clicks = []  # No clicks needed for admissions table
    headers=["#bec > div > div > header > h4"]
    council_table_selector = ["div.box table.table.table-striped"]
    council_data = scrape_headers_tables_by_selectors(council_url, clicks, headers, council_table_selector)
    #print(council_data)

    # #iic events
    iic_url="https://kmit.in/research/iic.php"
    clicks = [r"#Consultancy\ Projects > div:nth-child(2) > div > ul > li:nth-child(2) > a"]
    headers=[""]
    custom = [
        "iic events/workshops/training table"
    ]
    iic_table_selector = ["#events > table"]
    iic_table_data = scrape_customheaders_tables_by_selectors(iic_url, clicks, headers, iic_table_selector,custom_headers=custom)
    #print(iic_table_data)

    # # Patents 2022
    latestpatents_url="https://kmit.in/research/patents.php"
    clicks = [r"body > div > div > ul > li:nth-child(1) > a"]
    headers=[""]
    custom = [
        "Patents 2022 or patents 2022"
    ]
    patents_table_selector = ["#achieve2223 > div > div"]
    patents_2022_data = scrape_customheaders_tables_by_selectors(latestpatents_url, clicks, headers, patents_table_selector,custom_headers=custom)
    #print(patents_2022_data)

    # Patents 2021
    oldpatents_url="https://kmit.in/research/patents.php"
    clicks = [r"body > div > div > ul > li:nth-child(2) > a"]
    headers=[""]
    custom = [
        "Patents 2021 or patents 2021"
    ]
    oldpatents_table_selector = ["#achieve2122 > div > div"]
    patents_2021_data = scrape_customheaders_tables_by_selectors(oldpatents_url, clicks, headers, oldpatents_table_selector,custom_headers=custom)
    #print(patents_2021_data)

    # Patents(all)
    # patents_url="https://kmit.in/research/patents.php"
    # clicks = [r"body > div > div > ul > li:nth-child(3) > a"]
    # result = scrape_webpage(patents_url, clicks, heading_tag="h4", headless=True)
    # #print(result)

    # NSS Events(all)
    # nss_url="https://kmit.in/intiatives/nssevents.php"
    # clicks = [] # No clicks needed for admissions table
    # nss_table_selector = "#faculty-cse > div.container > div > div:nth-child(3) > div"
    # nss_data = scrape_webpage(nss_url, clicks, heading_tag="h4",headless=True)
    # #print(nss_data)

    # #NSS Events Contact Details
    # nss_url="https://kmit.in/intiatives/nssevents.php"
    # clicks = [] # No clicks needed for admissions table
    # headers=["#faculty-cse > div.container > div > div:nth-child(1) > header > h4"]
    # nss_details_table_selector = ["#faculty-cse > div.container > div > div:nth-child(3) > div > div > table"]
    # nss_detail_data = scrape_headers_tables_by_selectors(nss_url, clicks, headers, nss_details_table_selector)
    # #print(nss_detail_data)

    # aeb staff
    # aeb_url="https://kmit.in/examination/aebstaff.php"
    # clicks = [] # No clicks needed for admissions table
    # aeb_staff_detail_data = scrape_webpage(aeb_url, clicks, heading_tag="h4",headless=True)
    # #print(aeb_staff_detail_data)

    # sports activities
    # sports_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    # clicks = []  # No clicks needed
    # result = scrape_webpage(sports_url, clicks, heading_tag="h3", headless=True)
    # #print(result)

    #library
    # lib_url="https://kmit.in/infrastructure/aboutLib.php"
    # clicks = []
    # result = scrape_webpage(lib_url, clicks, heading_tag="h4", headless=True)
    # #print(result)

    #teleuniv
    # tel_url="https://kmit.in/uniqueness/teleuniv.php"
    # clicks = []
    # result = scrape_webpage(tel_url, clicks, heading_tag="h4", headless=True)
    # #print(result)

    #management(founder)
    foun_url = "https://kmit.in/administration/management.php"
    clicks = [r"#Sponsored\ Research > div:nth-child(2) > div > ul > li:nth-child(2) > a"]

    headers = [
        "#founder > div > div > div.col-sm-9 > header:nth-child(1) > h4",
        "#founder > div > div > div.col-sm-9 > header:nth-child(4) > h4"
    ]
    paragraphs = [
        ["#founder > div > div > div.col-sm-9 > p:nth-child(2)","#founder > div > div > div.col-sm-9 > p:nth-child(3)"],
        ["#founder > div > div > div.col-sm-9 > blockquote"]
    ]

    founder_data = scrape_headers_paragraphs_by_selectors(foun_url, clicks, headers, paragraphs)
    #print(founder_data)

    #management(director)
    dir_url = "https://kmit.in/administration/management.php"
    clicks = [r"#Sponsored\ Research > div:nth-child(2) > div > ul > li:nth-child(3) > a"]

    headers = [
        ""
    ]
    custom = [
        "Director of Genesis Solutions Pvt. Ltd."
    ]
    paragraphs = [
        ["#director > div > div > div.col-sm-9 > p"]
    ]

    director_data = scrape_customheaders_paragraphs_by_selectors(dir_url, clicks, headers, paragraphs,custom_headers=custom)
    #print(director_data)

    #sports(indoor)
    in_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    clicks = [r"#appsHeadingOne > a"]

    headers = [
        "#home-initiatives > div > div > div > div:nth-child(1) > header > h5"
    ]
    paragraphs = [
        ["#appsCollapseOne > div > p","#appsCollapseTwo > div > p","#appsCollapseThree > div > p"]
    ]

    in_data = scrape_headers_paragraphs_by_selectors(in_url, clicks, headers, paragraphs)
    #print(spo1_data)

    # #sports(yoga)
    # spo2_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    # clicks = [r"#appsHeadingTwo > a"]

    # headers = [
    #     "#home-initiatives > div > div > div > div:nth-child(1) > header > h5"
    # ]
    # paragraphs = [
    #     ["#appsCollapseTwo > div > p","#appsCollapseThree > div > p"]
    # ]

    # spo2_data = scrape_headers_paragraphs_by_selectors(spo2_url, clicks, headers, paragraphs)
    # #print(spo2_data)

    # #sports(chess,table tennis,carroms)
    # spo3_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    # clicks = [r"#appsHeadingThree > a"]

    # headers = [
    #     "#home-initiatives > div > div > div > div:nth-child(1) > header > h5"
    # ]
    # paragraphs = [
    #     ["#appsCollapseThree > div > p"]
    # ]

    # spo3_data = scrape_headers_paragraphs_by_selectors(spo3_url, clicks, headers, paragraphs)
    # #print(spo3_data)

    #sports(outdoor)
    out_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    clicks = [r"#headingOne > a"]

    headers = [
        "#home-initiatives > div > div > div > div:nth-child(2) > header > h5"
    ]
    paragraphs = [
        ["#collapseOne > div > p","#collapseTwo > div > p","#collapseThree > div > p"]
    ]

    out_data = scrape_headers_paragraphs_by_selectors(out_url, clicks, headers, paragraphs)
    #print(spo4_data)

    # #sports(basketball)
    # spo5_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    # clicks = [r"#headingTwo > a"]

    # headers = [
    #     "#home-initiatives > div > div > div > div:nth-child(2) > header > h5"
    # ]
    # paragraphs = [
    #     ["#collapseTwo > div > p","#collapseThree > div > p"]
    # ]

    # spo5_data = scrape_headers_paragraphs_by_selectors(spo5_url, clicks, headers, paragraphs)
    # #print(spo5_data)

    # #sports(volleyball)
    # spo6_url = "https://kmit.in/infrastructure/sportsfacilities.php"
    # clicks = [r"#headingThree > a"]

    # headers = [
    #     "#home-initiatives > div > div > div > div:nth-child(2) > header > h5"
    # ]
    # paragraphs = [
    #     ["#collapseThree > div > p"]
    # ]

    # spo6_data = scrape_headers_paragraphs_by_selectors(spo6_url, clicks, headers, paragraphs)
    # #print(spo6_data)

    #tessellator
    tess_url = "https://kmit.in/uniqueness/tessellator.php"
    clicks = []

    headers = [
        "body > div > section > div > header > h4"
    ]
    paragraphs = [
        ["body > div > section > div > div.row > div.col-sm-7 > p"]
    ]

    tess_data = scrape_headers_paragraphs_by_selectors(tess_url, clicks, headers, paragraphs)
    #print(tess_data)

    #lms
    lms_url = "https://kmit.in/uniqueness/lms.php"
    clicks = []

    headers = [
        "#about-hod > div > header > h4"
    ]
    paragraphs = [
        ["#about-hod > div > div > div:nth-child(1) > div:nth-child(2) > p:nth-child(1)","#about-hod > div > div > div:nth-child(2) > div:nth-child(2) > p:nth-child(1)","#about-hod > div > div > div:nth-child(3) > div:nth-child(2) > p:nth-child(1)"]
    ]

    lms_data = scrape_headers_paragraphs_by_selectors(lms_url, clicks, headers, paragraphs)
    #print(lms1_data)

    # #lms(drona app)
    # lms2_url = "https://kmit.in/uniqueness/lms.php"
    # clicks = []

    # headers = [
    #     "#about-hod > div > header > h4"
    # ]
    # paragraphs = [
    #     ["#about-hod > div > div > div:nth-child(2) > div:nth-child(2) > p:nth-child(1)","#about-hod > div > div > div:nth-child(3) > div:nth-child(2) > p:nth-child(1)"]
    # ]

    # lms2_data = scrape_headers_paragraphs_by_selectors(lms2_url, clicks, headers, paragraphs)
    # #print(lms2_data)

    # #lms(netra app)
    # lms3_url = "https://kmit.in/uniqueness/lms.php"
    # clicks = []

    # headers = [
    #     "#about-hod > div > header > h4"
    # ]
    # paragraphs = [
    #     ["#about-hod > div > div > div:nth-child(3) > div:nth-child(2) > p:nth-child(1)"]
    # ]

    # lms3_data = scrape_headers_paragraphs_by_selectors(lms3_url, clicks, headers, paragraphs)
    #print(lms3_data)

    # #director academic
    # diraca_url = "https://kmit.in/administration/academicdirector.php"
    # clicks = []

    # headers = [
    #     ""
    #     #"body > div > section > div > div.row > div.col-sm-9 > header:nth-child(3) > h4"
    # ]

    # custom = [
    #     "Academic Director"
    # ]

    # paragraphs = [
    #     "body > div > section > div > div.row > div.col-sm-9 > p"
    #     #["body > div > section > div > div.row > div.col-sm-9 > blockquote"]
    # ]

    # diraca_data = scrape_customheaders_paragraphs_by_selectors(diraca_url, clicks, headers, paragraphs,custom_headers=custom)
    #print(diraca_data)

    #management(president)
    pres_url = "https://kmit.in/administration/management.php"
    clicks = []

    headers = [
        ""
    ]
    custom = [
        "President of KMIT"
    ]
    paragraphs = [
        ["#president > div > div > div.col-sm-9 > p"]
    ]

    pres_data = scrape_customheaders_paragraphs_by_selectors(pres_url, clicks, headers, paragraphs,custom_headers=custom)
    #print(pres_data)

    #iic committee
    iic_url = "https://kmit.in/research/iic.php"
    clicks = []

    headers = [
        "#Consultancy\ Projects > div:nth-child(1) > header > h4"
    ]
    paragraphs = [
        ["#committee > div > ul > li:nth-child(1) > p","#committee > div > ul > li:nth-child(2) > p","#committee > div > ul > li:nth-child(3) > p","#committee > div > ul > li:nth-child(4) > p","#committee > div > ul > li:nth-child(5) > p","#committee > div > ul > li:nth-child(6) > p","#committee > div > ul > li:nth-child(7) > p","#committee > div > ul > li:nth-child(8) > p","#committee > div > ul > li:nth-child(9) > p"]
    ]

    iic_committee_data = scrape_headers_paragraphs_by_selectors(iic_url, clicks, headers, paragraphs)
    #print(iic_data)

    #vision and mission
    vis_url = "https://kmit.in/aboutus/aboutus.php"
    clicks = [r"body > div.background > div > ul > li:nth-child(2) > a"]

    headers = [
        "#visionmission > h6:nth-child(1)",
        "#visionmission > h6:nth-child(3)"
    ]
    paragraphs = [
        ["#visionmission > ul:nth-child(2) > li:nth-child(1) > p","#visionmission > ul:nth-child(2) > li:nth-child(2) > p"],
        ["#visionmission > ul:nth-child(4) > li:nth-child(1) > p","#visionmission > ul:nth-child(4) > li:nth-child(2) > p","#visionmission > ul:nth-child(4) > li:nth-child(3) > p","#visionmission > ul:nth-child(4) > li:nth-child(4) > p","#visionmission > ul:nth-child(4) > li:nth-child(5) > p","#visionmission > ul:nth-child(4) > li:nth-child(6) > p","#visionmission > ul:nth-child(4) > li:nth-child(7) > p"]
    ]

    vis_data = scrape_headers_paragraphs_by_selectors(vis_url, clicks, headers, paragraphs)
    #print(vis_data)


    # #aakarshan
    aakar_url = "https://kmit.in/intiatives/aakarshan.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]
    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    aakar_data = scrape_headers_paragraphs_by_selectors(aakar_url, clicks, headers, paragraphs)
    #print(aakar_data)

    #aakarshan(table)
    aakar_url = "https://kmit.in/intiatives/aakarshan.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "aakarshan club heads table"
    ]

    tables=["#bec > div > div > div > div > table"]

    aakar_table_data = scrape_customheaders_tables_by_selectors(aakar_url, clicks, headers, tables, custom_headers=custom)
    #print(aakar_table_data)

    #organising committee
    org_url = "https://kmit.in/intiatives/oc.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    org_data = scrape_headers_paragraphs_by_selectors(org_url, clicks, headers, paragraphs)
    #print(org_data)

    #organising committee(table)
    org_url = "https://kmit.in/intiatives/oc.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "organisation committee club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    org_table_data = scrape_customheaders_tables_by_selectors(org_url, clicks, headers, tables,custom_headers=custom)
    #print(org_table_data)

    #public relations
    pr_url = "https://kmit.in/intiatives/pr.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p"]
    ]

    pr_data = scrape_headers_paragraphs_by_selectors(pr_url, clicks, headers, paragraphs)
    #print(pr_data)


    #public relations(table)
    pr_url = "https://kmit.in/intiatives/pr.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "public relations club heads table"
    ]


    tables = ["#bec > div > div > div > div > table"]

    pr_table_data = scrape_customheaders_tables_by_selectors(pr_url, clicks, headers, tables,custom_headers=custom)
    #print(pr_table_data)

    #aalap
    aal_url = "https://kmit.in/intiatives/aalap.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    aal_data = scrape_headers_paragraphs_by_selectors(aal_url, clicks, headers, paragraphs)
    #print(aal_data)

    #aalap(table)
    aal_url = "https://kmit.in/intiatives/aalap.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "aalap club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    aal_table_data = scrape_customheaders_tables_by_selectors(aal_url, clicks, headers, tables,custom_headers=custom)
    #print(aal_table_data)


    #abhinaya
    abh_url = "https://kmit.in/intiatives/abhinaya.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    abh_data = scrape_headers_paragraphs_by_selectors(abh_url, clicks, headers, paragraphs)
    #print(abh_data)

    #abhinaya(table)
    abh_url = "https://kmit.in/intiatives/abhinaya.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "abhinaya club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    abh_table_data = scrape_customheaders_tables_by_selectors(abh_url, clicks, headers, tables,custom_headers=custom)
    #print(abh_table_data)

    #kaivalya
    kai_url = "https://kmit.in/intiatives/kaivalya.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    kai_data = scrape_headers_paragraphs_by_selectors(kai_url, clicks, headers, paragraphs)
    #print(kai_data)

    #kaivalya(tables)
    kai_url = "https://kmit.in/intiatives/kaivalya.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "kaivalya club heads table"
    ]


    tables = ["#bec > div > div > div > div > table"]

    kai_table_data = scrape_customheaders_tables_by_selectors(kai_url, clicks, headers, tables,custom_headers=custom)
    #print(kai_table_data)

    #kmitra(tables)
    kmi_url = "https://kmit.in/intiatives/kmitra.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "kmitra club heads table"
    ]

    tables = ["body > div > section > div > div > div.table-responsive > div > table"]

    kmi_table_data =  scrape_customheaders_tables_by_selectors(kmi_url, clicks, headers, tables,custom_headers=custom)
    #print(kmi_table_data)

    #kmitra
    kmi_url = "https://kmit.in/intiatives/kmitra.php"
    clicks = []

    headers = [
        "body > div > section > div > header > h4"
    ]

    paras = [
        ["body > div > section > div > div > div.col-md-7 > p:nth-child(1)"]
    ]

    kmi_data = scrape_headers_paragraphs_by_selectors(kmi_url, clicks, headers, paras)
    #print(kmi_data)

    #kreeda
    kre_url = "https://kmit.in/intiatives/kreeda.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paras = [
        ["#bec > div > div > p"]
    ]

    kre_data = scrape_headers_paragraphs_by_selectors(kre_url, clicks, headers, paras)
    #print(kre_data)

    # #kreeda(tables)
    kre_url = "https://kmit.in/intiatives/kreeda.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "kreeda club heads table"
    ]

    tables = ["#bec > div > div > ul > div > div > table"]

    kre_table_data = scrape_customheaders_tables_by_selectors(kre_url, clicks, headers, tables,custom_headers=custom)
    #print(kre_table_data)

    #mudra(tables)
    mudra_url = "https://kmit.in/intiatives/mudra.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "mudra club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    mudra_table_data = scrape_customheaders_tables_by_selectors(mudra_url, clicks, headers, tables,custom_headers=custom)
    #print(mudra_table_data)

    #mudra
    mudra_url = "https://kmit.in/intiatives/mudra.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    mudra_data = scrape_headers_paragraphs_by_selectors(mudra_url, clicks, headers, paragraphs)
    #print(mudra_data)

    #recurse
    recu_url = "https://kmit.in/intiatives/recurse.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    recu_data = scrape_headers_paragraphs_by_selectors(recu_url, clicks, headers, paragraphs)
    #print(recu_data)

    #recurse(tables)
    recu_url = "https://kmit.in/intiatives/recurse.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "recurse club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    recu_table_data = scrape_customheaders_tables_by_selectors(recu_url, clicks, headers, tables,custom_headers=custom)
    #print(recu_table_data)

    #photography(tables)
    tol_url = "https://kmit.in/intiatives/tol.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "traces of lenses-photography club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    tol_table_data = scrape_customheaders_tables_by_selectors(tol_url, clicks, headers, tables,custom_headers=custom)
    #print(tol_table_data)

    #photography
    tol_url = "https://kmit.in/intiatives/tol.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    tol_data = scrape_headers_paragraphs_by_selectors(tol_url, clicks, headers, paragraphs)
    #print(tol_data)

    #vachan
    vac_url = "https://kmit.in/intiatives/vachan.php"
    clicks = []

    headers = [
        "#bec > div > div > header > h4"
    ]

    paragraphs = [
        ["#bec > div > div > p:nth-child(2)"]
    ]

    vac_data = scrape_headers_paragraphs_by_selectors(vac_url, clicks, headers, paragraphs)
    #print(vac_data)

    #vachan(tables)
    vac_url = "https://kmit.in/intiatives/vachan.php"
    clicks = []

    headers = [
        ""
    ]

    custom = [
        "vachan club heads table"
    ]

    tables = ["#bec > div > div > div > div > table"]

    vac_table_data = scrape_customheaders_tables_by_selectors(vac_url, clicks, headers, tables,custom_headers=custom)
    #print(vac_table_data)

    #library
    lib_url = "https://kmit.in/infrastructure/aboutLib.php"
    clicks = []

    headers = [
        "#about-library > div > div > div > header > h4"
    ]

    paragraphs = [
        ["#about-library > div > div > div > p:nth-child(3)"]
    ]

    lib_data = scrape_headers_paragraphs_by_selectors(lib_url, clicks, headers, paragraphs)
    #print(lib_data)

    #library(staff table)
    libstaff_url = "https://kmit.in/infrastructure/aboutLib.php"
    clicks = []

    headers = [
        "#about-library > div > div > div > h5:nth-child(9)"
    ]

    tables = ["#about-library > div > div > div > div:nth-child(11) > div > table"]

    libstaff_data = scrape_headers_tables_by_selectors(libstaff_url, clicks, headers, tables)
    #print(libstaff_data)

    #principal
    pri_url = "https://kmit.in/administration/principal.php"
    clicks = []

    headers = [
        "body > div > section > div > header > h4"
    ]
    paragraphs = [
        ["body > div > section > div > div.row > div.col-sm-9 > p"]
    ]

    pri_data = scrape_headers_paragraphs_by_selectors(pri_url, clicks, headers, paragraphs)

    


    print("INFO: Data scraping finished.")

    # UPLOADING DATA TO CHROMADB
    print("INFO: Uploading data to ChromaDB...")
    upload_to_chromadb(placements_data, "placements_name_2024_2025","placements data - company hirings - 2024 to 2025 ")
    upload_to_chromadb(placements_data_2023_2024, "placements_name_2023_2024","placements data update - 2023 to 2024")
    upload_to_chromadb(admissions_data, "admission_fee_table", "admission fee - only contains details about admission procedures")
    upload_to_chromadb(courses_data, "course_details_table", "course data")
    upload_to_chromadb(cse_data, "cse_faculty_table", "cse data")
    upload_to_chromadb(csm_data, "csm_faculty_table", "csm data")
    upload_to_chromadb(it_data, "it_faculty_table", "it data")
    upload_to_chromadb(csd_data, "csd_faculty_table", "csd data")
    upload_to_chromadb(hs_data, "hs_faculty_table", "hs data")
    upload_to_chromadb(research_data, "research_details_table", "research data")
    upload_to_chromadb(contact_data, "contact_details_table", "contact data")
    upload_to_chromadb(council_data, "council_details_table", "council data")
    upload_to_chromadb(founder_data, "founder_details", "founder data")
    upload_to_chromadb(kre_data, "kreeda_details", "kreeda data")
    upload_to_chromadb(kre_table_data, "kreeda_table_details", "kreeda table data")
    upload_to_chromadb(mudra_data, "mudra_details", "mudra data")
    upload_to_chromadb(mudra_table_data, "mudra_table_details", "kreeda table data")
    upload_to_chromadb(aal_data, "aalap_details", "aalap data")
    upload_to_chromadb(aal_table_data, "aalap_table_details", "aalap table data")
    upload_to_chromadb(recu_table_data, "recurse_table_details", "recurse table data")
    upload_to_chromadb(recu_data, "recurse_details", "recurse data")
    upload_to_chromadb(aakar_data, "aakarshan_details", "aakarshan data")
    upload_to_chromadb(aakar_table_data, "aakarshan_table_details", "aakarshan table data")
    upload_to_chromadb(abh_data, "abhinaya_details", "abhinaya data")
    upload_to_chromadb(abh_table_data, "abhinaya_table_details", "abhinaya table data")
    upload_to_chromadb(kai_data, "kaivalya_details", "kaivalya data")
    upload_to_chromadb(kai_table_data, "kaivalya_table_details", "kaivalya table data")
    upload_to_chromadb(kmi_data, "kmitra_details", "kmitra data")
    upload_to_chromadb(kmi_table_data, "kmitra_table_details", "kmitra table data")
    upload_to_chromadb(tol_data, "tol_details", "tol data")
    upload_to_chromadb(tol_table_data, "tol_table_details", "tol details data")
    upload_to_chromadb(vac_data, "vachan_details", "vachan data")
    upload_to_chromadb(vac_table_data, "vachan_table_details", "vachan table data")
    upload_to_chromadb(lib_data, "library_details", "library data")
    upload_to_chromadb(libstaff_data, "library_staff_details", "library staff data")
    upload_to_chromadb(org_data, "organisation_details", "organisation data")
    upload_to_chromadb(org_table_data, "organisation_table_details", "organisation table data")
    upload_to_chromadb(pr_data, "pr_details", "pr data")
    upload_to_chromadb(pr_table_data, "pr_table_details", "pr table data")
    upload_to_chromadb(vis_data, "vision_details", "vision data")
    upload_to_chromadb(iic_table_data, "iic_events_details", "iic events data")
    upload_to_chromadb(iic_committee_data, "iic_committee_details", "iic committee data")
    upload_to_chromadb(pres_data, "pres_details", "pres data")
    upload_to_chromadb(diraca_data, "diraca_details", "diraca data")
    upload_to_chromadb(director_data, "director_details", "director data")
    upload_to_chromadb(lms_data, "lms_details", "lms data")
    upload_to_chromadb(in_data, "indoor_details", "indoor sports data")
    upload_to_chromadb(out_data, "outdoor_details", "outdoor sports data")
    upload_to_chromadb(tess_data, "tessellator_details", "tessellator data")
    upload_to_chromadb(patents_2021_data, "patents_2021_details", "patents 2021 data")
    upload_to_chromadb(patents_2022_data, "patents_2022_details", "patents 2022 data")
    upload_to_chromadb(pri_data, "principal_details", "principal data")
    print("INFO: ChromaDB upload finished.")

    # Start the background worker thread for Gemini API calls
    print("INFO: Starting Gemini API call worker thread...")
    api_worker_thread = threading.Thread(target=gemini_api_call_worker, daemon=True)
    api_worker_thread.start()

    # Run Flask app
    print(f"INFO: Starting Flask app on port 5000. Use threaded={True} for dev server with this queue model.")
    # For Flask's development server, `threaded=True` allows it to handle multiple
    # requests concurrently using threads, which is necessary for our event waiting model.
    # For production, use a proper WSGI server like Gunicorn with appropriate worker config.
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

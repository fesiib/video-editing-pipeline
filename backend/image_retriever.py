import requests
import os


def get_first_google_search_image(query, placeholder_image_url):
    query = query.strip()
    if len(query) == 0:
        print("WARNING: Empty query.")
        return placeholder_image_url
    
    # Replace with your API key and CSE ID
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    # Make the API request
    url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&searchType=image'
    response = requests.get(url)

    # Parse the JSON response
    if response.status_code == 200:
        data = response.json()
        # Extract the first image link
        if 'items' in data and len(data['items']) > 0:
            first_image_url = data['items'][0]['link']
            return first_image_url
        else:
            print ("WARNING: No image found.")
            return placeholder_image_url
    else:
        print("WARNING: API request failed.")
        return placeholder_image_url
    

def main():
    print(get_first_google_search_image("dog", "no"))

if __name__ == "__main__":
    main()
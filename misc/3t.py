import os
import requests

def setup_api():
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    API_URL = "https://w7k3ni2hylrqlds8.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json" 
    }
    return API_URL, headers

def query(API_URL, headers, message):
    payload = {
        "inputs": message,
        "parameters": {
            'max_new_tokens': 500,
            'min_new_tokens': 300,
            'return_full_text': False,
            'do_sample': True,
            'temperature': 0.7
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def chat():
    API_URL, headers = setup_api()
    print("Chat bot: Hello! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        user_input="""
write 'Conclusion' section for this blog :
From Ashes to Glory: Exploring the Phoenix Symbolism in Ukraine’s Coat of Arms

The Tryzub, Ukraine’s national coat of arms, is more than a mere emblem of statehood. It carries with it layers of historical, cultural, and symbolic meanings. One of the most intriguing aspects of this symbol is its connection to the phoenix — a mythical bird that rises from its ashes, symbolizing rebirth and resilience. This blog explores the rich tapestry of meanings embedded in the Tryzub, focusing on the phoenix symbolism that resonates deeply with Ukraine’s tumultuous history and enduring spirit.


The Historical Context of the Tryzub
The Tryzub, or trident, has ancient origins that date back over a thousand years to the Kyivan Rus era, where it served as a mark of power and statehood under the rule of Prince Volodymyr the Great. Throughout centuries of foreign domination and cultural oppression, the Tryzub endured as a symbol of the Ukrainian people’s unyielding spirit and desire for self-determination.

During the 20th century, Ukraine’s struggle for independence saw the re-emergence of the Tryzub as a national emblem, particularly during the Soviet era and the nationalist movements of the 1930s and 1940s. After Ukraine declared independence in 1991, the Tryzub was officially adopted as the national coat of arms, symbolizing the country’s resilience and sovereignty.

Phoenix Symbolism: A Universal Emblem of Rebirth
The phoenix is a legendary bird that has captured human imagination across cultures and epochs. It is often depicted as a bird that bursts into flames upon death, only to rise anew from its ashes. This cycle of death and rebirth makes the phoenix a potent symbol of transformation, resilience, and immortality.

In Ukrainian culture, the phoenix metaphor is intertwined with the nation’s historical narrative. It represents the ability of the Ukrainian people to withstand centuries of invasions, political upheaval, and attempts at cultural erasure, only to emerge stronger and more unified in their pursuit of independence and identity.

Romanticism, Surrealism, and Modernism: Evolving Interpretations
Art and literature have long used the phoenix to express human resilience and transformation. During the Romantic era, the phoenix symbolized individual struggle against societal constraints, while in Surrealism and Modernism, artists like Max Ernst and Gustave Moreau explored the bird’s enigmatic qualities to delve into themes of psychological transformation and the search for identity.

In contemporary Ukrainian culture, the phoenix continues to be a powerful symbol. Artistic expressions of the phoenix can be found in various forms, from visual arts and literature to public installations, reflecting the nation’s ongoing journey towards rebirth and self-realization amidst modern challenges.

The Tryzub: Phoenix Imagery in Design
The Tryzub’s design, with its three-pronged form, has been interpreted by some as an abstract representation of a rising phoenix, its wings outstretched, ready to take flight. This interpretation aligns with the emblem’s symbolic role as a beacon of resilience and renewal for Ukraine. The golden trident on a blue shield signifies hope and strength, much like the phoenix’s fiery transformation.

Case Studies: The Phoenix in Ukrainian Cultural Expressions
One of the most striking contemporary interpretations of the phoenix motif in Ukrainian culture is the “Phoenix” installation at the Burning Man festival. This art piece, resembling the Tryzub from certain angles, symbolized the rebirth of Ukrainian identity amid conflict and transformation. The burning of the structure served as a cathartic release, a symbolic cleansing of past traumas and a testament to the resilience of Ukrainian culture.

Another poignant example is the role of Cossack heritage in shaping modern interpretations of the phoenix symbol. During the Cossack era, the shift from hereditary nobility to a merit-based system echoed themes of rebirth and renewal, resonating with the phoenix’s symbolism of rising above adversity.

"""
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chat bot: Goodbye!")
            break
        
        response = query(API_URL, headers, user_input)
        #print(response)
        print("Chat bot:", response[0]['generated_text'])

if __name__ == "__main__":
    chat()
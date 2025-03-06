
import os
import streamlit as st
import openai
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class SimpleBlogContentGenerator:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the Blog Content Generator
        
        :param openai_api_key: OpenAI API key (optional, can use env variable)
        """
        # Set up OpenAI API
        self.openai_api_key = os.getenv('OPENAI_API_KEY') or openai_api_key
        if not self.openai_api_key:
            raise ValueError("OpenAI API Key is required")
        openai.api_key = self.openai_api_key
        
        # Simple storage for context (no vector DB)
        self.context_store = {}
        
        # Try to load previous context if exists
        try:
            if os.path.exists('context_store.json'):
                with open('context_store.json', 'r') as f:
                    self.context_store = json.load(f)
        except Exception as e:
            st.warning(f"Could not load previous context: {e}")
    
    def add_context(self, context: str):
        """
        Add context to simple storage
        
        :param context: Context text to store
        :return: Context ID
        """
        context_id = f"ctx_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.context_store[context_id] = {
            'text': context,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to file (optional, may not work in serverless)
        try:
            with open('context_store.json', 'w') as f:
                json.dump(self.context_store, f)
        except Exception as e:
            st.warning(f"Could not save context: {e}")
            
        return context_id
    
    def _generate_content(self, prompt: str, max_tokens: int = 1000):
        """
        Generate content using OpenAI GPT model
        
        :param prompt: Detailed prompt for content generation
        :param max_tokens: Maximum tokens for generation
        :return: Generated content
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful content generation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Content generation error: {e}")
            return "Unable to generate content due to an error."
    
    def generate_blog_content(self, keywords: list, context: str = None):
        """
        Generate complete blog content
        
        :param keywords: List of keywords for the blog
        :param context: Optional context 
        :return: Dictionary with blog title, description, and content
        """
        # Store context if provided
        context_id = None
        if context:
            context_id = self.add_context(context)
        
        # Generate title
        title_prompt = f"Generate a catchy blog title using these keywords: {', '.join(keywords)}"
        title = self._generate_content(title_prompt, max_tokens=50)
        
        # Generate description
        desc_prompt = f"Write a compelling meta description for a blog post about {', '.join(keywords)}"
        description = self._generate_content(desc_prompt, max_tokens=160)
        
        # Generate content with context if provided
        content_prompt = f"""
        Write a comprehensive blog post about {', '.join(keywords)}.
        Use an engaging and informative tone.
        {'Consider this additional context: ' + context if context else ''}
        Ensure the content is well-structured with clear headings and relevant to online diamond store "diamond sutra". do not give title.
        """
        content = self._generate_content(content_prompt, max_tokens=1000)
        
        return {
            "title": title,
            "description": description,
            "content": content
        }

def main():
    # Set page config
    st.set_page_config(
        page_title="Diamond Sutra SEO Optimised Blog",
        page_icon="✍️",
        layout="wide"
    )
    
    # Title and description
    st.title("💎 Diamond Sutra SEO Optimization")
    st.markdown("""
    Generate blog content using AI with keyword-based generation and optional context.
    
    ### How to Use:
    1. Enter keywords for your blog post
    2. (Optional) Provide additional context
    3. Click "Generate Content" to create your blog post
    """)
    
    # API Key input
    api_key = "api key"
    
    if not api_key and not os.getenv('OPENAI_API_KEY'):
        st.warning("Please enter your OpenAI API key in the sidebar to use this app.")
        return
        
    # Create generator instance
    try:
        generator = SimpleBlogContentGenerator(api_key)
        
        # Input section
        with st.form(key='blog_generation_form'):
            # Keywords input
            keywords = st.text_input(
                "Enter Keywords", 
                placeholder="e.g., Luxury Diamond Necklaces & Rings, Luxury Diamond Jewelry for Gifts, Hand Crafted Diamond Jewelry",
                help="Comma-separated keywords to guide content generation"
            )
            
            # Context input
            context = st.text_area(
                "Additional Context to Improve Blog Quality(Optional)", 
                placeholder="Paste any specific context or background information...",
                help="Provide additional details to refine content generation"
            )
            
            # Model selection
            model = "gpt-4"

            
            # Submit button
            submit_button = st.form_submit_button("Generate Content")
        
        # Content generation
        if submit_button:
            # Validate inputs
            if not keywords:
                st.warning("Please enter at least one keyword.")
                return
            
            # Show loading
            with st.spinner('Generating blog content...'):
                # Process keywords
                keyword_list = [k.strip() for k in keywords.split(',')]
                
                # Generate blog content
                try:
                    blog_output = generator.generate_blog_content(keyword_list, context)
                    
                    # Display results
                    st.success("Blog content generated successfully!")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3 = st.tabs(["Title & Description", "Blog Content", "Export"])
                    
                    with tab1:
                        # Title section
                        st.subheader("📝 Blog Title")
                        st.info(blog_output['title'])
                        
                        # Description section
                        st.subheader("📄 Meta Description")
                        st.info(blog_output['description'])
                    
                    with tab2:
                        # Content section
                        st.subheader("📖 Blog Content")
                        st.markdown(blog_output['content'])
                    
                    with tab3:
                        # Download options
                        st.subheader("💾 Export Options")
                        st.download_button(
                            label="Download as Text",
                            data=f"""Title: {blog_output['title']}

Description: {blog_output['description']}

Content:
{blog_output['content']}""",
                            file_name="ai_generated_blog.txt",
                            mime="text/plain"
                        )
                        
                        markdown_content = f"""# {blog_output['title']}

> {blog_output['description']}

{blog_output['content']}
"""
                        st.download_button(
                            label="Download as Markdown",
                            data=markdown_content,
                            file_name="ai_generated_blog.md",
                            mime="text/markdown"
                        )
                
                except Exception as e:
                    st.error(f"Error generating content: {e}")
    except Exception as e:
        st.error(f"Error initializing the application: {e}")

if __name__ == "__main__":
    main()

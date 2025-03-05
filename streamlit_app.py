import os
import streamlit as st
import openai
from dotenv import load_dotenv
import chromadb
import uuid

# Load environment variables
load_dotenv()

class BlogContentGenerator:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the Blog Content Generator
        
        :param openai_api_key: OpenAI API key (optional, can use env variable)
        """
        # Set up OpenAI API
        self.openai_api_key = "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"
        if not self.openai_api_key:
            raise ValueError("OpenAI API Key is required")
        openai.api_key = self.openai_api_key
        
        # Set up Vector Database with updated ChromaDB configuration
        self.chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="blog_context", 
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"Error creating collection: {e}")
            raise
    
    def add_context_to_vector_db(self, context: str):
        """
        Add context to vector database
        
        :param context: Text context to be vectorized
        :return: Confirmation of addition
        """
        # Generate a unique ID for the context
        context_id = str(uuid.uuid4())
        
        # Embed the context 
        embedding = self._get_embedding(context)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[context],
            ids=[context_id]
        )
        
        return context_id
    
    def _get_embedding(self, text: str, model="text-embedding-ada-002"):
        """
        Get embedding for a given text
        
        :param text: Input text
        :param model: Embedding model to use
        :return: Embedding vector
        """
        try:
            response = openai.Embedding.create(
                input=[text],
                model=model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            st.error(f"Embedding generation error: {e}")
            return None
    
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
        :param context: Optional context to be added to vector DB
        :return: Dictionary with blog title, description, and content
        """
        # Add context to vector DB if provided
        context_id = None
        retrieved_context = ""
        if context:
            context_id = self.add_context_to_vector_db(context)
            
            # Retrieve relevant context (if needed)
            try:
                results = self.collection.query(
                    query_embeddings=[self._get_embedding(" ".join(keywords))],
                    n_results=1
                )
                retrieved_context = results['documents'][0][0] if results['documents'] else ""
            except Exception as e:
                st.error(f"Context retrieval error: {e}")
        
        # Generate title
        title_prompt = f"Generate a catchy blog title using these keywords: {', '.join(keywords)}"
        title = self._generate_content(title_prompt, max_tokens=50)
        
        # Generate description
        desc_prompt = f"Write a compelling meta description for a blog post about {', '.join(keywords)}"
        description = self._generate_content(desc_prompt, max_tokens=160)
        
        # Generate content
        content_prompt = f"""
        Write a comprehensive blog post about {', '.join(keywords)}.
        Use an engaging and informative tone.
        {'Additional context: ' + retrieved_context if retrieved_context else ''}
        Ensure the content is well-structured with clear headings.
        """
        content = self._generate_content(content_prompt)
        
        return {
            "title": title,
            "description": description,
            "content": content
        }

def main():
    # Set page config
    st.set_page_config(
        page_title="AI Blog Content Generator",
        page_icon="✍️",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 AI Blog Content Generator")
    st.markdown("""
    Generate blog content using AI with keyword-based generation and optional context.
    
    ### How to Use:
    1. Enter keywords for your blog post
    2. (Optional) Provide additional context
    3. Click "Generate Content" to create your blog post
    """)
    
    # Create generator instance
    generator = BlogContentGenerator()
    
    # Input section
    with st.form(key='blog_generation_form'):
        # Keywords input
        keywords = st.text_input(
            "Enter Keywords", 
            placeholder="e.g., artificial intelligence, machine learning, data science",
            help="Comma-separated keywords to guide content generation"
        )
        
        # Context input
        context = st.text_area(
            "Additional Context (Optional)", 
            placeholder="Paste any specific context or background information...",
            help="Provide additional details to refine content generation"
        )
        
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
                
                # Title section
                st.subheader("📝 Blog Title")
                st.write(blog_output['title'])
                
                # Description section
                st.subheader("📄 Meta Description")
                st.write(blog_output['description'])
                
                # Content section
                st.subheader("📖 Blog Content")
                st.write(blog_output['content'])
                
                # Download options
                st.download_button(
                    label="Download Blog Content",
                    data=f"""Title: {blog_output['title']}

Description: {blog_output['description']}

Content:
{blog_output['content']}""",
                    file_name="ai_generated_blog.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"Error generating content: {e}")

if __name__ == "__main__":
    main()
from django import forms

class PostForm(forms.Form):
    QUERY = forms.CharField(max_length=50)
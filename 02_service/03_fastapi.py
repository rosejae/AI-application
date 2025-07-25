from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

"""
CRUD
Create, Read, Update, Delete
app.post: Create
app.get: Read  
app.put: Update
app.delete: Delete
"""

items = {
			0: {"name": "bread",
				"price": 1000},
		    1: {"name": "water",
			    "price": 500},
		    2: {"name": "라면",
				"price": 1200}
	    }

@app.get("/items")
def read_all_item():
    return items

# Path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int):
    item = items[item_id]
    return item

@app.get("/items/{item_id}/{key}")
def read_item_and_key(item_id: int, key: str):
    item = items[item_id][key]
    return item

# Query parameter
@app.get("/item-by-name")
def read_item_by_name(name: str):
    for item_id, item in items.items():
        if item['name'] == name:
            return item
    return {"message": "Not found"}

# Create, Post
class Item(BaseModel):
    name: str
    price: int

@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in items:
        return {"message": f"이미 item_id: {item_id}가 있습니다."}
    items[item_id] = item.model_dump()
    return {"message": "Success"}

# Update, Put
class ItemForUpdate(BaseModel):
    name: Optional[str]
    price: Optional[int]

@app.put("/items/{item_id}")
def update_item(item_id: int, item: ItemForUpdate):
    if item_id not in items:
        return {"message": f"item_id: {item_id}가 존재하지 않습니다."}
    if item.name:
        items[item_id]['name'] = item.name
    if item.price:
        items[item_id]['price'] = item.price
    return {"message": "Success"} 

# Delete, Delete
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in items:
        return {"message": f"item_id: {item_id}가 존재하지 않습니다."}
    items.pop(item_id)
    return {"message": "Success"}
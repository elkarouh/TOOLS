from hek_py2html import Builder, SELFCLOSING, Element, NOCONTENT
##########################################################################
from pscript import py2js
__all__ = ['HTMX', 'LOCAL','ParentElement']

GET="hx-get"
POST="hx-post"
PUT="hx-put"
PATCH="hx-patch"
DELETE="hx-delete"

class A(Element):
    def style(self):
        return super().style()+' A.style()'
class B(Element):
    def style(self):
        return super().style()+' B.style()'
class C(A, B):
    def style(self):
        return super().style()+' C.style()'

#a=C(Builder(),Builder())
#print(f"{a.style()=}")
#raise SystemExit

class ParentElement(Element):
    def __init__(self, xml, tagname, **attrs):
        Element.__init__(self,tagname,xml)
        super().__call__(**attrs)
class HTMX: # HTMX, requests go to a remote server
    def __init__(self, xml, text_content=NOCONTENT, routing_engine=None, model=None, **attrs):
        self.routing_engine=routing_engine
        self.model=model
        params = attrs
        if self.method():
            meth,link,*meth_params=self.method()
            params[meth]=link
            if meth_params:
                #print(f"{meth_params=}")
                params["hx-params"]=", ".join(str(param) for param in meth_params)
            if routing_engine:
                match meth:
                    case "hx-get":
                        METHOD='GET'
                        CALLBACK=self.process_get_request
                    case "hx-post":
                        METHOD='POST'
                        CALLBACK=self.process_post_request
                    case "hx-put":
                        METHOD='PUT'
                        CALLBACK=self.process_put_request
                    case "hx-patch":
                        METHOD='PATCH'
                        CALLBACK=self.process_patch_request
                    case "hx-delete":
                        METHOD='DELETE'
                        CALLBACK=self.process_delete_request
                    case _:
                        raise Exception(f"UNKNOWN METHOD: {meth}")
                # delete_previous_route
                routing_engine.routes = [route for route in routing_engine.routes 
                                              if not(route.method==METHOD and route.rule == link)]
                #print(f"REGISTERING ROUTING {CALLBACK.__name__} with method {METHOD:>6}: route {link}")
                routing_engine.route(link, method=METHOD, callback=CALLBACK)
        if self.trigger():
            params["hx-trigger"]=self.trigger()
        if self.target():
            params["hx-target"]=self.target()
        if self.swap():
            params["hx-swap"]=self.swap()
        if self.dialog():
            params["hx-confirm"]=self.dialog()
        if self.style():
            params["style"]=self.style()
        self.as_html(xml, **params)
    def as_html(self,xml,**params)  :
        raise NotImplementedError # PLEASE OVERRIDE
    def process_request(self):
        raise NotImplementedError
    def method(self):
        return
        return ("hx-get","/get_data")
        return ("hx-post","/post_data")
        return ("hx-put","/put_data", "OPTIONAL hx-params")
        return ("hx-patch","/patch_data")
        return ("hx-delete","/delete_data")
    def trigger(self):
        return
        # DEFAULT TRIGGERS
        # for input, textarea and select: "change", 
        # for form: "submit"
        # all the rest: "click"
        # to deviate from the default, specify e.g.
        return "changed"
        return "click[ctrlKey]"
        return "mouseenter"
        return "mouseenter once"
        return "keyup changed delay:500ms" # issue a request 500 milliseconds after a key up event if the input has been changed
        return "every 2s" # POLLING: Every 2 seconds, issue a request and load the response into the div
        return "load" # - fires once when the element is first loaded
        return "revealed" # - fires once when an element first scrolls into the viewport
        return "intersect" # - fires once when an element first intersects the viewport.
        # Multiple triggers can be specified in the hx-trigger attribute, separated by commas.
    def target(self):
        """
        Relative targets can be useful for creating flexible user interfaces without peppering your DOM with loads of id attributes.
        You can use the this keyword, which indicates that the element that the hx-target attribute is on is the target
        The closest <CSS selector> syntax will find the closest ancestor element or itself, that matches the given CSS selector.
            (e.g. closest tr will target the closest table row to the element)
        The next <CSS selector> syntax will find the next element in the DOM matching the given CSS selector.
        The previous <CSS selector> syntax will find the previous element in the DOM the given CSS selector.
        find <CSS selector> which will find the first child descendant element that matches the given CSS selector.
            (e.g find tr would target the first child descendant row to the element)
        """
        return
        # If you want the response to be loaded into a different element other than the one that made the request
        return "#parent-div"

    def swap(self):
        return
        return "innerHTML" # the default, puts the content INSIDE the target element
        return "outerHTML" # replaces the entire target element with the returned content
        return "afterbegin" # prepends the content before the first child INSIDE the target
        return "beforeend" # appends the content after the last child INSIDE the target
        return "beforebegin" # prepends the content before the target in the target's parent element
        return "afterend" # appends the content AFTER the target in the target's parent element
        return "none" # does not append content from response (Out of Band Swaps and Response Headers will still be processed)
    def dialog(self):
        return ""
        return "Are you sure?" # special value "unset" to suppress inherited confirms !!!!
    def style(self):
        return ""

class LOCAL(Element): # ALPINE.JS client-side
    def __init__(self, xml, text_content=NOCONTENT, **attrs):
        Element.__init__(self,attrs.get('tag','div'),xml)
        if not xml.root:
            xml.root=self
        params={}
        if attrs:
            params['x-data']="{"+",".join( f"{item[0]}:{item[1]}" for item in attrs.items() if item[0] not in ('id','class_'))+"}"
            if 'id' in attrs:
                params['id']=attrs['id']
            if 'class_' in attrs:
                params['class_']=attrs['class_']
        if self.init():
            params['x-init']=self.init()
        if self.text_content():
            params['x-text']=self.text_content()
        if self.on_click():
            params['x-on:click']=self.on_click()
        if self.on_click_outside():
            params['x-on:click.outside']=self.on_click_outside()
        if self.on_mouseenter():
            params['x-on:mouseenter']=self.on_mouseenter()
        if self.observes():
            observer,observed=self.observes()
            params[f'x-bind:observer']=observed
        if self.show_when():
            params['x-show']=self.show_when()
        if self.style():
            params["style"]=self.style() # todo, replace by class when using tailwind_css
        super().__call__(text_content, **params)
    def init(self):
        return ""
    def style(self):
        return ""
        #return super().style()+' LOCAL style'
    def observes(self):
        return "" # return (observer,observed)
        # observer is mostly an attribute
        # But it can also be a class
        return ("class","open ? '' : 'hidden'")
    def show_when(self):
        return ""
    def text_content(self):
        return ""
    def on_click(self):
        return ""
    def on_click_outside(self):
        return ""
    def on_mouseenter(self):
        return ""

if __name__=="__main__":
    class MyHTMX(HTMX0):
        def trigger(self):
            return "mouseenter"
        def method(self):
            return ("hx-get","/get_data")
        def target(self):
            # If you want the response to be loaded into a different element other than the one that made the request
            return "#parent-div"
        def swap(self):
            return "outerHTML" # replaces the entire target element with the returned content

    class MyInput(HTMX0):
        def __init__(self, xml, text_content=NOCONTENT, **kwargs):
            kwargs['tag']="input"
            kwargs['class_']="btn btn-primary"
            super().__init__(xml, text_content=text_content, type="text", name="q", placeholder="Search...", **kwargs)
        def trigger(self):
            return "keyup changed delay:500ms"
        def method(self):
            return ("hx-get","/trigger_delay")
        def target(self):
            # If you want the response to be loaded into a different element other than the one that made the request
            return "#search-results"
        def swap(self):
            return "innerHTML" # default behaviour
    class MyFORM(HTMX0):
        def __init__(self,xml,**kwargs):
            kwargs['tag']="form"
            kwargs['class_']="btn btn-primary"
            super().__init__(xml,text_content=NOCONTENT,**kwargs) # a parent has no text content
        def method(self):
            return ("hx-post","/submit")
        def target(self):
            return "#new-book"
        def swap(self):
            return "beforeend"
    class MyForm2(HTMX):
        def __init__(self,xml,text_content=NOCONTENT,**kwargs):
            super().__init__(xml,text_content,**kwargs) # a parent has no text content
        def render_html(self,xml,**kwargs):
            xml.form()
        def method(self):
            return ("hx-post","/tasks/")
        def target(self):
            return "body"         
    class MyDeleteWidget(HTMX0):
        def __init__(self,xml, text_content, routing_engine, model, task_id, **kwargs):
            self.task_id=task_id
            kwargs['tag']="a"
            super().__init__(xml, text_content, routing_engine=routing_engine, model=model, **kwargs)    
        def method(self):
            return ("hx-delete",f"/tasks/{self.task_id}")
        def swap(self):
            return "outerHTML"
        def target(self):
            return "body"
    class MyStyle(Element):
        def style(self):
            return "---------inherited style"
            #return super().style()+"---------style"
    class MyLOCAL(LOCAL,MyStyle):
        def __init__(self,xml,*args,**kwargs):
            kwargs['tag']="input"
            #MyStyle.__init__(self,row_number)
            MyStyle.__init__(self,"",xml)
            LOCAL.__init__(self,xml,*args,**kwargs)
        def init(self):
            return "a=[]"
        def text_content(self):
            return "count"
        def on_click(self):
            return "count++"
        def on_mouseenter(self):
            return "count--"
        def show_when(self):
            return "open"
        def style(self):
            return super().style()+" this_is_my_style"
    with Builder() as xml: # pylon CSS
        xml.write_doctype("html") # look at https://swlkr.com/ridgecss/examples.html
        with xml.html(lang="en",dir="ltr"):
            with xml.head:
                xml.link(rel="stylesheet", href="https://unpkg.com/pyloncss@latest/css/pylon.css")
                xml.link(rel="stylesheet", media="(prefers-color-scheme: light), (prefers-color-scheme: none)", href="ridge-light.css")
                xml.link(rel="stylesheet", media="(prefers-color-scheme: dark)", href="ridge-dark.css")
                xml.link(rel="stylesheet", href="ridge.css")
            with xml.body:
                with xml.hstack(spacing="s"): # spacing is xxs, xs, s, m, l ,xl
                    xml.input(SELFCLOSING,type="text", placeholder="Your name")
                    xml.input(SELFCLOSING,type="email", placeholder="Your email")
                    xml.button("Submit")
                with xml.vstack(spacing="s"):
                    xml.input(SELFCLOSING,type="text", placeholder="Your name")
                    xml.input(SELFCLOSING,type="email", placeholder="Your email")
                    xml.button("Submit")
                # align-x is left, center , right
                # align-y is left, center , right
                xml.vstack("Hello world.",**{"stretch":True, "align-x":"center", "align-y":"center"})
                with xml.list(spacing="xs"):
                    # Lists are similar to a vstack but have some built in conveniences.
                    # The List element will assume its immediate children are row elements
                    # and draw borders between them while omitting the last row.
                    xml.text("Red")
                    xml.text("Green")
                    xml.text("Blue")
                with xml.list(spacing="s"):
                    with xml.hstack(spacing="xs"):
                        xml.input(SELFCLOSING,type="checkbox")
                        xml.label("Do laundry")
                        xml.spacer()
                        xml.div("Due today")
                    with xml.hstack(spacing="xs"):
                        xml.input(SELFCLOSING,type="checkbox")
                        xml.label("Buy milk")
                with xml.hstack(spacing="xs"):
                    xml.input(stretch=True, type="text", placeholder="Add a todo")
                    xml.button("Add")
                    xml.divider() # Dividers are used to visually divide items in a stack.
                    xml.button("View all")
                with xml.hstack(debug='true'):
                    xml.button("Cancel")
                    xml.spacer()
                    xml.button("Submit")
                with xml.hstack(debug=True):
                    xml.button("Cancel")
                    xml.spacer()
                    xml.button("Submit")
                MyHTMX(xml,"HELLO",a=8,b=9)
                MyLOCAL(xml,"SALAM",count=8,b=9,tag='input')
                with MyLOCAL(xml, class_="blue", id="my_id"):
                    with xml.hstack(debug='true'):
                        xml.button("Cancel")
                        xml.button("Submitjesss\\",'stretch')
                xml.input(SELFCLOSING,type="text", placeholder="Book Title", name="title", class_="form-control mb-3")
                xml.input(SELFCLOSING,type="text", placeholder="Book Author", name="author", class_="form-control mb-3")
                xml.button("Submit",type="submit", class_="btn btn-primary")
                xml["my-component"](SELFCLOSING,name="Alex") # CUSTOM COMPONENT !!!!
                with MyInput(xml, id="my_id"):
                    xml.spacer()
                with MyFORM(xml, id="my_id"):
                    xml.spacer()
                MyDeleteWidget(xml, 'X', None, None, 555)        
                MyForm2(xml, "HELLO")
                with MyForm2(xml):
                    xml.input(SELFCLOSING, id="my_id2",type="text", name="title", placeholder="New task", value="", autocomplete="off")
                    xml.button("Add",type="submit", value="Add", class_="button-primary", role="button")


    print(str(xml))
